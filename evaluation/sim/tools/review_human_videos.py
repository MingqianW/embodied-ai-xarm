"""Serve a localhost-only blinded video-review UI for a review manifest."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import threading
from typing import Any
from urllib.parse import unquote
from urllib.parse import urlparse

from evaluation.sim.human_review import FAILURE_REASONS  # noqa: E402
from evaluation.sim.human_review import PRIMARY_LABELS  # noqa: E402
from evaluation.sim.human_review import load_decisions  # noqa: E402
from evaluation.sim.human_review import reviewer_items  # noqa: E402
from evaluation.sim.human_review import save_decision  # noqa: E402
from evaluation.sim.outputs import read_json  # noqa: E402

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Blinded xArm video review</title>
<style>
body { font-family: sans-serif; max-width: 980px; margin: 2rem auto; color: #1f2937; }
video { width: 100%; max-height: 620px; background: #111827; }
button, select, textarea { font: inherit; margin: .3rem; padding: .45rem; }
button { cursor: pointer; } .hidden { display: none; } #message { font-weight: bold; }
</style></head><body>
<h1>Blinded xArm video review</h1>
<p id="status"></p><p><strong id="review-id"></strong></p><p id="prompt"></p>
<video id="video" controls autoplay muted playsinline></video>
<section id="form">
<p><button onclick="choose('SUCCESS')">Success [S]</button>
<button onclick="choose('FAILURE')">Failure [F]</button>
<button onclick="choose('UNCERTAIN')">Uncertain [U]</button></p>
<p>Selected label: <strong id="label">none</strong></p>
<label id="reason-wrap" class="hidden">Failure reason (optional):
<select id="reason"><option value="">No subtype</option>{reason_options}</select></label>
<p><label>Notes (optional):<br><textarea id="notes" rows="3" cols="80"></textarea></label></p>
<button onclick="submitReview()">Save decision and continue</button>
</section>
<p id="message"></p>
<script>
let current = null; let selected = null;
const statusEl = document.getElementById('status');
const message = document.getElementById('message');
function choose(label) { selected = label; document.getElementById('label').textContent = label;
  document.getElementById('reason-wrap').classList.toggle('hidden', label !== 'FAILURE'); }
async function loadNext() {
  const response = await fetch('/api/next'); const data = await response.json();
  if (data.complete) { current = null; document.getElementById('form').classList.add('hidden');
    document.getElementById('video').removeAttribute('src'); document.getElementById('review-id').textContent = 'Review complete';
    document.getElementById('prompt').textContent = ''; statusEl.textContent = `${data.reviewed}/${data.total} decisions saved.`; return; }
  current = data; selected = null; document.getElementById('form').classList.remove('hidden');
  document.getElementById('review-id').textContent = data.review_id;
  document.getElementById('prompt').textContent = data.prompt;
  statusEl.textContent = `${data.reviewed}/${data.total} decisions saved.`;
  document.getElementById('label').textContent = 'none'; document.getElementById('reason').value = '';
  document.getElementById('notes').value = ''; document.getElementById('reason-wrap').classList.add('hidden');
  const video = document.getElementById('video'); video.src = data.video_url; video.load(); message.textContent = '';
}
async function submitReview() {
  if (!current || !selected) { message.textContent = 'Choose SUCCESS, FAILURE, or UNCERTAIN first.'; return; }
  const response = await fetch('/api/decision', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({
    review_id: current.review_id, human_label: selected, human_failure_reason: document.getElementById('reason').value,
    notes: document.getElementById('notes').value})});
  const data = await response.json(); if (!response.ok) { message.textContent = data.error || 'Unable to save decision.'; return; }
  await loadNext();
}
document.addEventListener('keydown', event => { if (event.target.matches('textarea,select')) return;
  if (event.key.toLowerCase() === 's') choose('SUCCESS');
  if (event.key.toLowerCase() === 'f') choose('FAILURE');
  if (event.key.toLowerCase() === 'u') choose('UNCERTAIN');
});
loadNext();
</script></body></html>"""


class ReviewApplication:
    def __init__(self, *, review_root: Path, allow_overwrite_decisions: bool) -> None:
        self.review_root = Path(review_root).expanduser().resolve()
        self.private_manifest = read_json(self.review_root / "manifest_private.json")
        self.items = reviewer_items(self.private_manifest)
        self.order = [str(item["review_id"]) for item in self.private_manifest["items"]]
        self.decisions_path = self.review_root / "human_review.csv"
        self.allow_overwrite_decisions = allow_overwrite_decisions
        self.lock = threading.Lock()
        decisions = load_decisions(self.decisions_path)
        unknown = sorted(set(decisions).difference(self.items))
        if unknown:
            raise ValueError(f"Review CSV contains IDs outside this manifest: {unknown}")

    def next_item(self) -> dict[str, Any]:
        with self.lock:
            decisions = load_decisions(self.decisions_path)
            next_id = next((review_id for review_id in self.order if review_id not in decisions), None)
        if next_id is None:
            return {"complete": True, "reviewed": len(decisions), "total": len(self.order)}
        item = self.items[next_id]
        return {
            "complete": False,
            "review_id": next_id,
            "prompt": item["prompt"],
            "video_url": f"/video/{next_id}",
            "reviewed": len(decisions),
            "total": len(self.order),
        }

    def submit(self, payload: dict[str, Any]) -> dict[str, str]:
        with self.lock:
            return save_decision(
                csv_path=self.decisions_path,
                review_ids=set(self.items),
                review_id=str(payload.get("review_id", "")),
                label=str(payload.get("human_label", "")),
                failure_reason=str(payload.get("human_failure_reason", "")),
                notes=str(payload.get("notes", "")),
                allow_overwrite=self.allow_overwrite_decisions,
            )

    def video_path(self, review_id: str) -> Path:
        try:
            path = Path(self.items[review_id]["video_path"])
        except KeyError as exc:
            raise KeyError(f"Unknown review ID: {review_id}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Video is unavailable for {review_id}: {path}")
        return path


def _handler(application: ReviewApplication) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: HTTPStatus, value: dict[str, Any]) -> None:
            encoded = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                options = "".join(f'<option value="{reason}">{reason}</option>' for reason in FAILURE_REASONS)
                encoded = PAGE.replace("{reason_options}", options).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
                return
            if path == "/api/next":
                self._send_json(HTTPStatus.OK, application.next_item())
                return
            if path == "/api/status":
                item = application.next_item()
                self._send_json(
                    HTTPStatus.OK,
                    {"reviewed": item["reviewed"], "total": item["total"], "complete": item["complete"]},
                )
                return
            if path.startswith("/video/"):
                try:
                    video = application.video_path(unquote(path.removeprefix("/video/")))
                    content_type = mimetypes.guess_type(video.name)[0] or "application/octet-stream"
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(video.stat().st_size))
                    self.end_headers()
                    with video.open("rb") as handle:
                        while chunk := handle.read(1024 * 1024):
                            self.wfile.write(chunk)
                except (FileNotFoundError, KeyError) as exc:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/decision":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if not 0 < content_length <= 16_384:
                    raise ValueError("Decision payload size is invalid")
                payload = json.loads(self.rfile.read(content_length))
                if not isinstance(payload, dict):
                    raise ValueError("Decision payload must be a JSON object")
                self._send_json(HTTPStatus.OK, application.submit(payload))
            except FileExistsError as exc:
                self._send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def log_message(self, format: str, *args: object) -> None:
            del format, args  # Avoid logging reviewer behavior to terminal history.

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--allow-overwrite-decisions", action="store_true")
    args = parser.parse_args()
    if args.host != "127.0.0.1":
        raise ValueError("Human review UI intentionally binds only to 127.0.0.1")
    application = ReviewApplication(
        review_root=args.review_root, allow_overwrite_decisions=args.allow_overwrite_decisions
    )
    server = ThreadingHTTPServer((args.host, args.port), _handler(application))
    print(f"Blinded review UI: http://{args.host}:{args.port}")
    print(f"Allowed primary labels: {', '.join(PRIMARY_LABELS)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReview server stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

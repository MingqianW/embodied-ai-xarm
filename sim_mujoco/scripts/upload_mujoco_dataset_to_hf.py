"""Safely inspect or upload a prepared MuJoCo dataset to Hugging Face."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim_mujoco.paths import mujoco_output_root


DEFAULT_LOCAL_DIR = mujoco_output_root() / "hf_ready" / "xarm_mujoco_red_block_v1"
DEFAULT_REPO_ID = "MingqianW/xarm_mujoco_red_block_v1"
IGNORE_PATTERNS = (
    ".git/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*.tmp",
    "**/*.lock",
)


def _ignored(relative: str) -> bool:
    value = relative.replace("\\", "/")
    return any(fnmatch.fnmatch(value, pattern) for pattern in IGNORE_PATTERNS)


def _local_files(local_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    files: list[dict[str, Any]] = []
    ignored: list[str] = []
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(local_dir).as_posix()
        if _ignored(relative):
            ignored.append(relative)
            continue
        files.append({"path": relative, "size_bytes": path.stat().st_size})
    return files, ignored


def _validate_local(local_dir: Path) -> dict[str, Any]:
    required = (
        "README.md",
        "DATASET_CARD.md",
        "UPLOAD_PLAN.md",
        "MANIFEST.json",
        "meta/info.json",
        "meta/tasks.jsonl",
        "meta/episodes.jsonl",
        "meta/episodes_stats.jsonl",
    )
    missing = [relative for relative in required if not (local_dir / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"HF-ready directory is missing: {missing}")
    info = json.loads(
        (local_dir / "meta" / "info.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (local_dir / "MANIFEST.json").read_text(encoding="utf-8")
    )
    if info.get("codebase_version") != "v2.1":
        raise ValueError("HF-ready LeRobot codebase_version must be v2.1")
    if info.get("splits") != {
        "train": f"0:{int(info.get('total_episodes', -1))}"
    }:
        raise ValueError("HF-ready split metadata is inconsistent")
    if manifest.get("repo_id") is None or not isinstance(
        manifest.get("files"),
        list,
    ):
        raise ValueError("Invalid MANIFEST.json")
    seen: set[str] = set()
    for row in manifest["files"]:
        if not isinstance(row, dict):
            raise ValueError("MANIFEST file entry must be an object")
        relative = str(row.get("path", ""))
        if not relative or relative in seen:
            raise ValueError(f"Invalid or duplicate manifest path: {relative!r}")
        seen.add(relative)
        path = (local_dir / relative).resolve()
        try:
            path.relative_to(local_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"Manifest path escapes local directory: {relative}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(row.get("size_bytes", -1)):
            raise ValueError(f"Manifest size mismatch: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != row.get("sha256"):
            raise ValueError(f"Manifest SHA256 mismatch: {relative}")
    return {
        "codebase_version": info["codebase_version"],
        "total_episodes": int(info["total_episodes"]),
        "total_frames": int(info["total_frames"]),
        "manifest_repo_id": manifest["repo_id"],
        "manifest_file_entries": len(manifest["files"]),
    }


def _repo_status(repo_id: str) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError:
        return {
            "exists": None,
            "authenticated_check": False,
            "detail": "huggingface_hub is not installed",
        }
    api = HfApi()
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="dataset")
    except Exception as exc:
        name = type(exc).__name__
        if name in {"RepositoryNotFoundError", "HfHubHTTPError"}:
            # A private repository can also look absent without authentication,
            # so preserve that uncertainty rather than claiming it is missing.
            try:
                api.whoami()
            except Exception:
                return {
                    "exists": None,
                    "authenticated_check": False,
                    "detail": (
                        f"{name}; authenticate with `hf auth login` or "
                        "`huggingface-cli login` to distinguish missing/private"
                    ),
                }
            return {
                "exists": False,
                "authenticated_check": True,
                "detail": name,
            }
        return {
            "exists": None,
            "authenticated_check": False,
            "detail": f"{name}: {exc}",
        }
    return {
        "exists": True,
        "authenticated_check": True,
        "detail": getattr(info, "id", repo_id),
    }


def build_upload_plan(
    local_dir: Path,
    *,
    repo_id: str,
    private: bool,
    commit_message: str,
    repo_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    local_dir = local_dir.resolve()
    if not local_dir.is_dir():
        raise FileNotFoundError(local_dir)
    consistency = _validate_local(local_dir)
    if consistency["manifest_repo_id"] != repo_id:
        raise ValueError(
            f"MANIFEST repo_id={consistency['manifest_repo_id']!r} does not "
            f"match proposed repo_id={repo_id!r}"
        )
    files, ignored = _local_files(local_dir)
    if not files:
        raise ValueError("No uploadable files")
    return {
        "dry_run": True,
        "local_dir": str(local_dir),
        "proposed_repo_id": repo_id,
        "repo_type": "dataset",
        "private": bool(private),
        "commit_message": commit_message,
        "total_file_count": len(files),
        "total_size_bytes": sum(row["size_bytes"] for row in files),
        "files_to_upload": files,
        "ignored_files": ignored,
        "remote_repository": repo_status or _repo_status(repo_id),
        "local_metadata_consistent": True,
        "local_consistency": consistency,
        "upload_strategy": (
            "HfApi.upload_large_folder; resumable local cache; "
            "no remote deletion"
        ),
    }


def _remote_manifest_matches(api: Any, repo_id: str, local_manifest: dict[str, Any]) -> bool:
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=repo_id,
            filename="MANIFEST.json",
            repo_type="dataset",
        )
        remote = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        remote.get("repo_id") == local_manifest.get("repo_id")
        and remote.get("dataset_identity") == local_manifest.get("dataset_identity")
    )


def _upload(args: argparse.Namespace, plan: dict[str, Any]) -> dict[str, Any]:
    if args.dry_run:
        raise ValueError("Actual upload refuses --dry-run")
    if not args.upload or not args.yes:
        raise ValueError("Actual upload requires both --upload and --yes")
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Install/authenticate huggingface_hub before actual upload"
        ) from exc
    api = HfApi()
    try:
        api.whoami()
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face authentication is required; run `hf auth login` "
            "or `huggingface-cli login`"
        ) from exc
    status = _repo_status(args.repo_id)
    local_manifest = json.loads(
        (args.local_dir / "MANIFEST.json").read_text(encoding="utf-8")
    )
    if status.get("exists") is True and not _remote_manifest_matches(
        api,
        args.repo_id,
        local_manifest,
    ):
        raise RuntimeError(
            "Refusing to upload into an existing repository whose "
            "MANIFEST.json identity does not match this dataset"
        )
    print(
        f"FINAL UPLOAD TARGET: {args.repo_id} "
        f"({plan['total_file_count']} files)"
    )
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    values = {
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "folder_path": str(args.local_dir.resolve()),
        "commit_message": args.commit_message,
        "allow_patterns": ["**"],
        "ignore_patterns": list(IGNORE_PATTERNS),
    }
    signature = inspect.signature(api.upload_large_folder)
    accepted = {
        key: value
        for key, value in values.items()
        if key in signature.parameters
    }
    api.upload_large_folder(**accepted)
    return {
        **plan,
        "dry_run": False,
        "uploaded": True,
        "remote_repository": _repo_status(args.repo_id),
        "remote_deletions": 0,
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.upload and args.dry_run:
        raise ValueError("--upload and --dry-run are mutually exclusive")
    if args.yes and not args.upload:
        raise ValueError("--yes has no effect without --upload")
    status = _repo_status(args.repo_id)
    plan = build_upload_plan(
        args.local_dir,
        repo_id=args.repo_id,
        private=args.private,
        commit_message=args.commit_message,
        repo_status=status,
    )
    if not args.upload:
        print(json.dumps(plan, indent=2))
        return plan
    result = _upload(args, plan)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--commit-message",
        default="Add MuJoCo red-block scripted-oracle dataset v1",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicitly request the default no-upload inspection mode.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Opt out of the default dry run and request an actual upload.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Second mandatory confirmation for actual upload.",
    )
    args = parser.parse_args()
    execute(args)


if __name__ == "__main__":
    main()

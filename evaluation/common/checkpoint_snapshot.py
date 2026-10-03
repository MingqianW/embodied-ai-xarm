"""Download and verify a fixed public checkpoint snapshot outside source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request


def verify_file(path: Path, metadata: dict) -> dict:
    size = path.stat().st_size
    if size != metadata["size"]:
        raise ValueError(f"Snapshot size mismatch: {path}")
    lfs = metadata.get("lfs")
    digest = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        digest.update(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    expected = lfs["sha256"] if lfs else metadata["blobId"]
    if actual != expected:
        raise ValueError(f"Snapshot content hash mismatch: {path}")
    return {"file": metadata["rfilename"], "size": size, "digest": actual, "algorithm": "sha256" if lfs else "git_blob_sha1"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("A fixed 40-character Hugging Face commit revision is required")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo_id):
        raise ValueError("Invalid repository identifier")
    directory = args.directory.expanduser().resolve()
    url = f"https://huggingface.co/api/models/{args.repo_id}/revision/{args.revision}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        metadata = json.load(response)
    if metadata["sha"] != args.revision:
        raise ValueError("Hugging Face revision mismatch")
    identity = {"repo_id": args.repo_id, "revision": args.revision}
    marker = directory / "_snapshot_identity.json"
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError("Refusing to reuse a directory belonging to another snapshot")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError("Refusing to download over an unowned nonempty directory")
        directory.mkdir(parents=True, exist_ok=True)
        with marker.open("x", encoding="utf-8") as stream:
            json.dump(identity, stream)

    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=args.repo_id, revision=args.revision, local_dir=directory)
    checks = []
    for row in metadata["siblings"]:
        path = (directory / row["rfilename"]).resolve()
        if directory not in path.parents:
            raise ValueError("Snapshot file path escapes its root")
        checks.append(verify_file(path, row))

    # Read every logical value through OCDBT, so missing/checksum-invalid data
    # referenced by manifests fail here, before allocating a GPU to restore.
    import tensorstore as ts

    store = ts.KvStore.open({"driver": "ocdbt", "base": {"driver": "file", "path": str(directory / "params") + "/"}}).result()
    keys = store.list().result()
    if not keys:
        raise ValueError("OCDBT parameter store has no keys")
    bytes_read = 0
    for key in keys:
        bytes_read += len(store.read(key).result().value)
    report = {
        **identity, "directory": str(directory), "verified": True,
        "files": checks, "ocdbt_keys_read": len(keys), "ocdbt_bytes_read": bytes_read,
        "model_restored": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()

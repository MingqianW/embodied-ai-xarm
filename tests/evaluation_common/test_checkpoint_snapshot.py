import hashlib
from pathlib import Path

import pytest

from evaluation.common.checkpoint_snapshot import verify_file


@pytest.mark.parametrize("lfs", [True, False])
def test_snapshot_verifies_lfs_and_git_blob_content(tmp_path: Path, lfs: bool) -> None:
    contents = b"checkpoint fixture"
    path = tmp_path / "fixture"
    path.write_bytes(contents)
    metadata = {"rfilename": "fixture", "size": len(contents)}
    if lfs:
        metadata["lfs"] = {"sha256": hashlib.sha256(contents).hexdigest()}
    else:
        metadata["blobId"] = hashlib.sha1(f"blob {len(contents)}\0".encode() + contents).hexdigest()
    assert verify_file(path, metadata)["size"] == len(contents)
    path.write_bytes(b"x" * len(contents))
    with pytest.raises(ValueError, match="content hash"):
        verify_file(path, metadata)
    path.write_bytes(b"short")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_file(path, metadata)

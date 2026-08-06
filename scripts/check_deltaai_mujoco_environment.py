"""Check MuJoCo/DeltaAI readiness without contacting an OpenPI server."""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
import tempfile
from importlib import import_module, metadata
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim_mujoco.paths import (
    active_model_path,
    camera_config_path,
    mujoco_dataset_root,
    mujoco_output_root,
    repository_root,
    task_config_path,
)


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _result(name: str, status: str, detail: Any) -> dict[str, Any]:
    return {"name": name, "status": status, "detail": detail}


def run_checks(*, require_egl: bool = False) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    root = repository_root()
    model_path = active_model_path()
    camera_path = camera_config_path()
    task_path = task_config_path()

    version_ok = sys.version_info[:2] == (3, 11)
    checks.append(
        _result(
            "python",
            "passed" if version_ok else "failed",
            {"version": platform.python_version(), "executable": sys.executable},
        )
    )
    machine = platform.machine().lower()
    target_arch = machine in {"aarch64", "arm64"}
    checks.append(
        _result(
            "architecture",
            "passed" if target_arch else "warning",
            {"machine": machine, "target": "aarch64"},
        )
    )
    backend = os.environ.get("MUJOCO_GL", "")
    backend_ok = backend.lower() == "egl"
    checks.append(
        _result(
            "opengl_backend",
            "passed" if backend_ok else ("failed" if require_egl else "warning"),
            {"MUJOCO_GL": backend or "(unset)", "target": "egl"},
        )
    )

    for name, path in {
        "repository_root": root,
        "active_mjcf": model_path,
        "camera_config": camera_path,
        "task_config": task_path,
    }.items():
        checks.append(
            _result(name, "passed" if path.exists() else "failed", str(path))
        )

    mesh_dir = (
        root
        / "third_party"
        / "xarm_ros2"
        / "xarm_description"
        / "meshes"
        / "xarm6"
        / "visual"
    )
    meshes = [mesh_dir / ("link_base.stl" if index == 0 else f"link{index}.stl") for index in range(7)]
    missing_meshes = [str(path) for path in meshes if not path.is_file()]
    checks.append(
        _result(
            "required_meshes",
            "passed" if not missing_meshes else "failed",
            {"count": len(meshes) - len(missing_meshes), "missing": missing_meshes},
        )
    )

    for distribution in (
        "numpy",
        "opencv-python-headless",
        "Pillow",
        "imageio",
        "imageio-ffmpeg",
        "lerobot",
        "datasets",
        "huggingface-hub",
        "PyYAML",
    ):
        value = _version(distribution)
        checks.append(
            _result(
                f"package:{distribution}",
                "passed" if value else "failed",
                value or "not installed",
            )
        )

    try:
        import numpy as np
        from PIL import Image

        sample = np.zeros((16, 16, 3), dtype=np.uint8)
        encoded = io.BytesIO()
        Image.fromarray(sample).save(encoded, format="PNG")
        if not encoded.getvalue().startswith(b"\x89PNG"):
            raise RuntimeError("PNG signature missing")
        checks.append(_result("image_encoding", "passed", "Pillow PNG"))
    except Exception as exc:
        checks.append(_result("image_encoding", "failed", f"{type(exc).__name__}: {exc}"))

    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=64, width=64)
        renderer.update_scene(data, camera="base_camera")
        image = renderer.render()
        close = getattr(renderer, "close", None)
        if close is not None:
            close()
        valid = image.shape == (64, 64, 3) and str(image.dtype) == "uint8"
        checks.append(
            _result(
                "mujoco_render",
                "passed" if valid else "failed",
                {
                    "mujoco_version": getattr(mujoco, "__version__", "unknown"),
                    "shape": list(image.shape),
                    "dtype": str(image.dtype),
                    "backend": backend or "platform default",
                },
            )
        )
    except Exception as exc:
        checks.append(_result("mujoco_render", "failed", f"{type(exc).__name__}: {exc}"))

    for module_name in ("lerobot", "datasets", "huggingface_hub"):
        try:
            module = import_module(module_name)
            checks.append(_result(f"import:{module_name}", "passed", str(module.__file__)))
        except Exception as exc:
            checks.append(
                _result(f"import:{module_name}", "failed", f"{type(exc).__name__}: {exc}")
            )

    for name, directory in {
        "output_root": mujoco_output_root(),
        "dataset_root": mujoco_dataset_root(),
    }.items():
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".write_check_", dir=directory):
                pass
            checks.append(_result(f"writable:{name}", "passed", str(directory)))
        except Exception as exc:
            checks.append(
                _result(f"writable:{name}", "failed", f"{type(exc).__name__}: {exc}")
            )

    try:
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="mujoco_video_check_") as temporary:
            path = Path(temporary) / "test.mp4"
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (64, 64),
            )
            opened = writer.isOpened()
            if opened:
                writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
            writer.release()
            valid = opened and path.is_file() and path.stat().st_size > 0
        checks.append(
            _result(
                "video_writer",
                "passed" if valid else "warning",
                "MP4/mp4v available" if valid else "MP4 unavailable; PNG fallback will be used",
            )
        )
    except Exception as exc:
        checks.append(
            _result(
                "video_writer",
                "warning",
                f"{type(exc).__name__}: {exc}; PNG fallback will be used",
            )
        )

    failed = [item["name"] for item in checks if item["status"] == "failed"]
    warnings = [item["name"] for item in checks if item["status"] == "warning"]
    return {
        "ready": not failed,
        "failed": failed,
        "warnings": warnings,
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-egl",
        action="store_true",
        help="Fail unless MUJOCO_GL=egl and the render check passes.",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    report = run_checks(require_egl=args.require_egl)
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")
    raise SystemExit(0 if report["ready"] else 1)


if __name__ == "__main__":
    main()

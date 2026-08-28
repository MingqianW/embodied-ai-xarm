#!/usr/bin/env python3
"""Prepare the runtime-only split-pad friction model and formal protocol."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from sim_mujoco.scripts.run_friction_ablation import (  # noqa: E402
    FROZEN_MODEL,
    FROZEN_MODEL_SHA256,
    PAD_FRICTION_A,
)
from sim_mujoco.scripts.run_scripted_gripper_slip_experiments import (  # noqa: E402
    _model_invariant_hashes,
)


ALLOWED_ROOT = Path("/work/nvme/bfmk/mw89")
CAMERA_CONFIG = (
    PROJECT_ROOT
    / "sim_mujoco/config/diagnostics/legacy_split_pad_camera_calibration.yaml"
)
SOURCE_PROTOCOL = PROJECT_ROOT / "sim_mujoco/config/formal_xarm_pi05_eval_v2.json"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _target_friction(lower: float, upper: float) -> dict[str, float]:
    values = (lower, upper)
    if not all(np.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("Lower and upper sliding friction must be positive and finite")
    return {
        "left_fingertip_pad": float(lower),
        "left_fingertip_pad_upper": float(upper),
        "right_fingertip_pad": float(lower),
        "right_fingertip_pad_upper": float(upper),
    }


def _variant_xml(target_friction: dict[str, float]) -> bytes:
    if _sha256(FROZEN_MODEL) != FROZEN_MODEL_SHA256:
        raise RuntimeError("Frozen split-pad source SHA-256 changed")
    root = ET.parse(FROZEN_MODEL).getroot()
    seen: set[str] = set()
    for geom in root.findall(".//geom"):
        name = geom.get("name")
        if name not in target_friction:
            continue
        actual = [float(value) for value in geom.get("friction", "").split()]
        expected = [PAD_FRICTION_A[name], 0.02, 0.002]
        if not np.allclose(actual, expected, atol=1e-12):
            raise RuntimeError(f"Unexpected source friction: {name}={actual}")
        geom.set("friction", f"{target_friction[name]:g} 0.02 0.002")
        seen.add(name)
    if seen != set(target_friction):
        raise RuntimeError(
            f"Missing source pads: {sorted(set(target_friction) - seen)}"
        )
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def validate_variant(lower: float = 0.7, upper: float = 0.6) -> dict[str, Any]:
    target_friction = _target_friction(lower, upper)
    xml = _variant_xml(target_friction)
    baseline = mujoco.MjModel.from_xml_path(str(FROZEN_MODEL))
    candidate = mujoco.MjModel.from_xml_string(xml.decode("utf-8"))
    before = _model_invariant_hashes(baseline)
    after = _model_invariant_hashes(candidate)
    changed = sorted(key for key in before if before[key] != after[key])
    expected_changed = (
        []
        if all(np.isclose(value, 2.0) for value in target_friction.values())
        else ["geom_friction"]
    )
    if changed != expected_changed:
        raise RuntimeError(f"Variant changed forbidden compiled fields: {changed}")
    effective: dict[str, list[float]] = {}
    for name in target_friction:
        geom_id = mujoco.mj_name2id(candidate, mujoco.mjtObj.mjOBJ_GEOM, name)
        effective[name] = candidate.geom_friction[geom_id].tolist()
    return {
        "passed": True,
        "source_model": str(FROZEN_MODEL),
        "source_sha256": FROZEN_MODEL_SHA256,
        "variant_xml_sha256": _sha256_bytes(xml),
        "changed_compiled_invariants": changed,
        "effective_pad_friction": effective,
        "requested_pad_sliding_friction": target_friction,
        "variant_xml": xml,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--evaluation-output-root", type=Path)
    parser.add_argument("--lower-friction", type=float, default=0.7)
    parser.add_argument("--upper-friction", type=float, default=0.6)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def _safe_child(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == ALLOWED_ROOT or ALLOWED_ROOT not in resolved.parents:
        raise ValueError(f"Path must be below {ALLOWED_ROOT}: {resolved}")
    return resolved


def main() -> None:
    args = _parser().parse_args()
    validation = validate_variant(args.lower_friction, args.upper_friction)
    xml = validation.pop("variant_xml")
    if args.validate_only:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return
    if args.output_dir is None or args.evaluation_output_root is None:
        raise ValueError("--output-dir and --evaluation-output-root are required")
    output = _safe_child(args.output_dir)
    evaluation_output = _safe_child(args.evaluation_output_root)
    if output.exists():
        raise FileExistsError(f"Refusing existing variant output: {output}")
    if evaluation_output.exists():
        raise FileExistsError(
            f"Refusing existing evaluation output: {evaluation_output}"
        )
    output.mkdir(parents=True, exist_ok=False)
    model_path = output / "split_pad_selected_friction.xml"
    model_path.write_bytes(xml)
    if _sha256(model_path) != validation["variant_xml_sha256"]:
        raise RuntimeError("Written runtime model failed SHA-256 validation")

    protocol = json.loads(SOURCE_PROTOCOL.read_text(encoding="utf-8"))
    protocol["paths"]["camera_config"] = str(CAMERA_CONFIG)
    protocol["paths"]["robot_xml"] = str(model_path)
    protocol_path = output / "formal_v2_runtime_friction.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "xarm_runtime_friction_policy_variant_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        **validation,
        "runtime_model": str(model_path),
        "runtime_model_sha256": _sha256(model_path),
        "camera_config": str(CAMERA_CONFIG),
        "camera_config_sha256": _sha256(CAMERA_CONFIG),
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "evaluation_output_root": str(evaluation_output),
        "production_mjcf_modified": False,
    }
    manifest_path = output / "variant_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

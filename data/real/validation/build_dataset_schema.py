"""Build the compact real-data contract required by strict sim validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _vectors(table, name: str) -> np.ndarray:
    values = table[name].combine_chunks().to_pylist()
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 7:
        raise ValueError(f"{name} must be Nx7, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    files = sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No canonical episode parquet files: {dataset_dir}")

    states = []
    actions = []
    for path in files:
        table = pq.read_table(path, columns=["state", "actions"])
        states.append(_vectors(table, "state"))
        actions.append(_vectors(table, "actions"))
    state = np.concatenate(states)
    action = np.concatenate(actions)

    report = {
        "dataset_identity": {
            "path": str(dataset_dir),
            "hub_metadata_verified": False,
        },
        "episodes": len(files),
        "frames": len(state),
        "state_action_contract": {
            "order": [
                "joint_1_rad",
                "joint_2_rad",
                "joint_3_rad",
                "joint_4_rad",
                "joint_5_rad",
                "joint_6_rad",
                "gripper_mm",
            ],
            "gripper_observed_real_range": [
                float(min(state[:, 6].min(), action[:, 6].min())),
                float(max(state[:, 6].max(), action[:, 6].max())),
            ],
            "runtime_safety_range": [50.0, 845.0],
            "raw_action_semantics": "next-frame absolute target",
        },
        "observed_real_state_distribution": {
            "minimum": state.min(axis=0).tolist(),
            "maximum": state.max(axis=0).tolist(),
        },
        "observed_real_action_distribution": {
            "minimum": action.min(axis=0).tolist(),
            "maximum": action.max(axis=0).tolist(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "episodes": report["episodes"],
                "frames": report["frames"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

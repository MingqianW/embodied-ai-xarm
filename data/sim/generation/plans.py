"""Supported simulation dataset plans and exact authorized output roots."""

from __future__ import annotations

from pathlib import Path


DATASET_PLANS = {
    "xarm_mujoco_clean_multitask_stable_v3": {
        "counts": {
            "red_pepper": 50,
            "blue_block": 25,
            "red_block": 25,
            "smallest_block": 25,
            "largest_block": 25,
            "place_red_pepper_in_ring": 50,
        },
        "roots": frozenset(
            {
                Path("/work/nvme/bfmk/mw89/mujoco_datasets/raw/xarm_mujoco_clean_multitask_stable_v3"),
                Path("/work/nvme/bfmk/mw89/mujoco_datasets/local/xarm_mujoco_clean_multitask_stable_v3"),
                Path("/work/nvme/bfmk/mw89/mujoco_datasets/smoke/xarm_mujoco_clean_multitask_stable_v3"),
                Path("/work/nvme/bfmk/mw89/logs/xarm_mujoco_clean_multitask_stable_v3"),
            }
        ),
    },
    "xarm_mujoco_clean_multitask_stable_v4_10x_real": {
        "counts": {
            "red_pepper": 500,
            "blue_block": 240,
            "red_block": 250,
            "smallest_block": 240,
            "largest_block": 250,
            "place_red_pepper_in_ring": 500,
        },
        "roots": frozenset(
            {
                Path("/work/nvme/bfmk/mw89/mujoco_datasets/raw/xarm_mujoco_clean_multitask_stable_v4_10x_real"),
                Path("/work/nvme/bfmk/mw89/mujoco_datasets/local/xarm_mujoco_clean_multitask_stable_v4_10x_real"),
                Path("/work/nvme/bfmk/mw89/mujoco_datasets/smoke/xarm_mujoco_clean_multitask_stable_v4_10x_real"),
                Path("/work/nvme/bfmk/mw89/logs/xarm_mujoco_clean_multitask_stable_v4_10x_real"),
            }
        ),
    },
}


def _plan(dataset_version: str) -> dict:
    try:
        return DATASET_PLANS[dataset_version]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported simulation dataset version: {dataset_version}"
        ) from exc


def expected_counts(dataset_version: str) -> dict[str, int]:
    return dict(_plan(dataset_version)["counts"])


def expected_roots(dataset_version: str) -> frozenset[Path]:
    # ``Path('/work/...')`` is drive-relative when interpreted on Windows,
    # while configured paths are normalized by ``Path.resolve``.  Normalize
    # both sides so the exact-root safety check is platform-independent.
    return frozenset(
        root.resolve(strict=False) for root in _plan(dataset_version)["roots"]
    )


AUTHORIZED_ROOTS = frozenset(
    root.resolve(strict=False)
    for plan in DATASET_PLANS.values()
    for root in plan["roots"]
)

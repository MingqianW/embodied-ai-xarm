"""Produce one real OpenPI π0.5 xArm batch from a local LeRobot dataset."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENPI_ROOT = PROJECT_ROOT / "third_party" / "openpi"
OPENPI_SOURCE = OPENPI_ROOT / "src"


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _shape_dtype(value: Any) -> dict[str, Any]:
    array = _array(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
    }


def smoke(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = args.dataset_dir.resolve()
    if not (dataset_dir / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset: {dataset_dir}")
    if str(OPENPI_SOURCE) not in sys.path:
        sys.path.insert(0, str(OPENPI_SOURCE))

    # OpenPI's loader resolves local datasets as HF_LEROBOT_HOME / repo_id.
    # Use the existing directory directly; no copying, linking, or Hub access.
    os.environ["HF_LEROBOT_HOME"] = str(dataset_dir.parent)
    loader_repo_id = dataset_dir.name

    from typing_extensions import override

    import openpi.models.model as model
    import openpi.models.pi0_config as pi0_config
    import openpi.policies.libero_policy as libero_policy
    import openpi.training.config as config
    import openpi.training.data_loader as data_loader
    import openpi.transforms as transforms

    @dataclasses.dataclass(frozen=True)
    class LeRobotXArmDataConfig(config.DataConfigFactory):
        use_delta_joint_actions: bool = True

        @override
        def create(
            self,
            assets_dirs: Path,
            model_config: model.BaseModelConfig,
        ) -> config.DataConfig:
            repack = transforms.Group(
                inputs=[
                    transforms.RepackTransform(
                        {
                            "observation/image": "image",
                            "observation/wrist_image": "wrist_image",
                            "observation/state": "state",
                            "actions": "actions",
                            "prompt": "prompt",
                        }
                    )
                ]
            )
            data_transforms = transforms.Group(
                inputs=[
                    libero_policy.LiberoInputs(
                        model_type=model_config.model_type
                    )
                ],
                outputs=[libero_policy.LiberoOutputs()],
            )
            if self.use_delta_joint_actions:
                delta_mask = transforms.make_bool_mask(6, -1)
                data_transforms = data_transforms.push(
                    inputs=[transforms.DeltaActions(delta_mask)],
                    outputs=[transforms.AbsoluteActions(delta_mask)],
                )
            model_transforms = config.ModelTransformFactory()(model_config)
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=repack,
                data_transforms=data_transforms,
                model_transforms=model_transforms,
                action_sequence_keys=("actions",),
            )

    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=10,
        discrete_state_input=False,
    )
    train_config = config.TrainConfig(
        name="pi05_xarm_mujoco_batch_smoke",
        exp_name="local_validation",
        model=model_config,
        data=LeRobotXArmDataConfig(
            repo_id=loader_repo_id,
            base_config=config.DataConfig(prompt_from_task=True),
        ),
        assets_base_dir=str(args.assets_dir.resolve()),
        checkpoint_base_dir=str(args.assets_dir.resolve() / "unused_checkpoints"),
        batch_size=args.batch_size,
        num_workers=0,
        num_train_steps=1,
        wandb_enabled=False,
    )
    loader = data_loader.create_data_loader(
        train_config,
        shuffle=False,
        num_batches=1,
        skip_norm_stats=True,
        framework="pytorch",
    )
    observation, actions = next(iter(loader))
    image_shapes = {
        key: _shape_dtype(value)
        for key, value in observation.images.items()
    }
    result = {
        "passed": True,
        "requested_repo_id": args.repo_id,
        "loader_repo_id": loader_repo_id,
        "dataset_dir": str(dataset_dir),
        "openpi_source": str(OPENPI_SOURCE),
        "model_type": str(model_config.model_type),
        "model_action_dim": model_config.action_dim,
        "model_action_horizon": model_config.action_horizon,
        "images": image_shapes,
        "image_masks": {
            key: _shape_dtype(value)
            for key, value in observation.image_masks.items()
        },
        "state": _shape_dtype(observation.state),
        "tokenized_prompt": _shape_dtype(observation.tokenized_prompt),
        "tokenized_prompt_mask": _shape_dtype(
            observation.tokenized_prompt_mask
        ),
        "actions": _shape_dtype(actions),
    }
    expected_images = {
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    }
    if set(observation.images) != expected_images:
        raise ValueError(f"Unexpected image keys: {set(observation.images)}")
    expected_image_shape = [args.batch_size, 224, 224, 3]
    if any(
        value["shape"] != expected_image_shape
        for value in image_shapes.values()
    ):
        raise ValueError(f"Unexpected image batch shapes: {image_shapes}")
    if result["state"]["shape"] != [args.batch_size, 32]:
        raise ValueError(f"Unexpected padded state shape: {result['state']}")
    if result["actions"]["shape"] != [args.batch_size, 10, 32]:
        raise ValueError(f"Unexpected padded action shape: {result['actions']}")
    expected_tokens = [args.batch_size, model_config.max_token_len]
    if result["tokenized_prompt"]["shape"] != expected_tokens:
        raise ValueError(
            f"Unexpected tokenized prompt shape: "
            f"{result['tokenized_prompt']}"
        )
    if not all(
        descriptor["finite"]
        for descriptor in [
            *image_shapes.values(),
            result["state"],
            result["actions"],
            result["tokenized_prompt"],
        ]
    ):
        raise ValueError("OpenPI batch contains NaN or Inf")
    padded_state = _array(observation.state)
    padded_actions = _array(actions)
    if not np.allclose(padded_state[:, 7:], 0.0):
        raise ValueError("State dimensions 7:32 are not zero-padded")
    if not np.allclose(padded_actions[:, :, 7:], 0.0):
        raise ValueError("Action dimensions 7:32 are not zero-padded")

    # Independently load the first 10-action chunk and prove that OpenPI made
    # only joints 0:6 relative while leaving the gripper absolute.
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    direct = LeRobotDataset(
        loader_repo_id,
        root=dataset_dir,
        delta_timestamps={
            "actions": [index / 10.0 for index in range(10)]
        },
        download_videos=False,
    )
    first = direct[0]
    source_state = _array(first["state"]).astype(np.float32)
    source_actions = _array(first["actions"]).astype(np.float32)
    expected_delta = source_actions.copy()
    expected_delta[:, :6] -= source_state[:6]
    np.testing.assert_allclose(
        padded_state[0, :7],
        source_state,
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        padded_actions[0, :, :7],
        expected_delta,
        rtol=0.0,
        atol=1e-6,
    )
    result["transform_semantics"] = {
        "joint_dimensions_0_to_5": "delta relative to observation state",
        "gripper_dimension_6": "absolute",
        "dimensions_7_to_31": "zero padding",
        "verified_against_direct_lerobot_sample": True,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=PROJECT_ROOT / "sim_mujoco" / "output" / "openpi_smoke_assets",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    smoke(args)


if __name__ == "__main__":
    main()

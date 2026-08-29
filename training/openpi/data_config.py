"""Project xArm DataConfig factory, created lazily against an OpenPI install."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from training.datasets.adapter import XARM_ACTION_SEQUENCE_KEYS, XARM_OPENPI_REPACK_MAPPING


def make_xarm_data_config_class(config: Any, model: Any, libero_policy: Any, transforms: Any) -> type:
    """Return the repository's small xArm adapter subclass for this OpenPI API."""

    @dataclasses.dataclass(frozen=True)
    class LeRobotXArmDataConfig(config.DataConfigFactory):
        use_delta_joint_actions: bool = True

        def create(self, assets_dirs: Path, model_config: Any) -> Any:
            repack = transforms.Group(
                inputs=[transforms.RepackTransform(XARM_OPENPI_REPACK_MAPPING)]
            )
            data_transforms = transforms.Group(
                inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
                outputs=[libero_policy.LiberoOutputs()],
            )
            if self.use_delta_joint_actions:
                delta_mask = transforms.make_bool_mask(6, -1)
                data_transforms = data_transforms.push(
                    inputs=[transforms.DeltaActions(delta_mask)],
                    outputs=[transforms.AbsoluteActions(delta_mask)],
                )
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=repack,
                data_transforms=data_transforms,
                model_transforms=config.ModelTransformFactory()(model_config),
                action_sequence_keys=XARM_ACTION_SEQUENCE_KEYS,
            )

    return LeRobotXArmDataConfig

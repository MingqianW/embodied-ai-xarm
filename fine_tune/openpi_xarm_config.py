"""
OpenPI config snippet for xArm LeRobot fine-tuning.

Copy the class and config entry into Physical-Intelligence/openpi's
src/openpi/training/config.py, then replace `local/xarm_pi05_data`
with the repo id used by fine_tune/convert_xarm_raw_to_lerobot.py.
"""

# Add these imports in openpi/src/openpi/training/config.py if they are not present:
# import dataclasses
# import pathlib
# from typing_extensions import override
# import openpi.transforms as _transforms
# import openpi.models.model as _model
# import openpi.policies.libero_policy as libero_policy


XARM_CONFIG_SNIPPET = r'''
@dataclasses.dataclass(frozen=True)
class LeRobotXArmDataConfig(DataConfigFactory):
    """Data config for this repository's xArm LeRobot dataset.

    Expected LeRobot frame keys:
        image, wrist_image, state, actions, task

    `state` and `actions` are 7D:
        [j1, j2, j3, j4, j5, j6, gripper_mm]

    The converter stores actions as the next absolute joint/gripper state. For
    pi0/pi0.5 training, we convert the first 6 joint action dimensions to deltas
    and leave the gripper dimension absolute.
    """

    use_delta_joint_actions: bool = True

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Maps LeRobot dataset keys to the common OpenPI training/inference keys.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
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

        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("actions",),
        )


# Add these TrainConfig entries to _CONFIGS. Start with
# pi05_xarm_colab_smoke to verify data loading and checkpointing, then use
# pi05_xarm_full_finetune for full pi0.5 fine-tuning.
#
# Full fine-tuning intentionally does NOT set LoRA variants and does NOT set a
# freeze_filter. This follows OpenPI's pi05_libero full fine-tuning pattern.
TrainConfig(
    name="pi05_xarm_full_finetune",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=10,
        discrete_state_input=False,
    ),
    data=LeRobotXArmDataConfig(
        repo_id="local/xarm_pi05_data",
        base_config=DataConfig(prompt_from_task=True),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    batch_size=16,
    num_train_steps=30_000,
    save_interval=5_000,
    wandb_enabled=True,
),

TrainConfig(
    name="pi05_xarm",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=10,
        discrete_state_input=False,
    ),
    data=LeRobotXArmDataConfig(
        repo_id="local/xarm_pi05_data",
        base_config=DataConfig(prompt_from_task=True),
    ),
    # lr_schedule=_optimizer.CosineDecaySchedule(
    #     warmup_steps=1_000,
    #     peak_lr=5e-5,
    #     decay_steps=1_000_000,
    #     decay_lr=5e-5,
    # ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=0.999,
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    batch_size=16,
    num_train_steps=20_001,
    save_interval=5000,
    wandb_enabled=True,
),

TrainConfig(
    name="pi05_xarm_colab_smoke",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotXArmDataConfig(
        repo_id="local/xarm_pi05_data",
        base_config=DataConfig(prompt_from_task=True),
    ),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=100,
        peak_lr=5e-5,
        decay_steps=10_000,
        decay_lr=5e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    freeze_filter=pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=10,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    ema_decay=None,
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    batch_size=16,
    num_train_steps=1_000,
    save_interval=500,
    wandb_enabled=False,
),
'''


if __name__ == "__main__":
    print(XARM_CONFIG_SNIPPET)



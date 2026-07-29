# Required MuJoCo Migration Files

Files: 101
All present: True
All tracked: True

## Simulation Assets

| Path | Tracked | Generated | Portable | Size (KiB) | SHA256 |
| --- | --- | --- | --- | ---: | --- |
| `sim_mujoco/assets/xarm6/xarm6_arm.xml` | True | True | True | 5.10 | `16902a4e5a9a205ea76440c3c6abd3accc1494706a4fd087c8641a934b3fc6cf` |
| `sim_mujoco/assets/xarm6/xarm6_pick_scene.xml` | True | True | True | 18.93 | `9f6d8ef3dc3be7ed8996dfc8513742e6257322efaafe6268536b639f4e80edd4` |
| `sim_mujoco/scenes/minimal.xml` | True | False | True | 0.80 | `8c8dcbd36b3c0206dfd7c4bf99d1003f043234b62922c377821ae8b4c0c37731` |
| `sim_mujoco/config/camera_calibration.yaml` | True | False | True | 1.17 | `0ec030ef4bc28d9155deb663a82c1132d2ff6f1bf16af6ef6a8a68b5089350f4` |
| `sim_mujoco/calibration/baseline_camera_calibration.yaml` | True | False | True | 0.73 | `c3a22d6f04138ea46683f3374bf3fe9b3f77af5f44581cb98e61a6478fbb037c` |
| `sim_mujoco/config/task_scenes.yaml` | True | False | True | 4.94 | `6a341b24975c2ee5413b12f7d6495eafd32b938e898d1db092e922f15ecd6b1e` |
| `third_party/xarm_ros2/xarm_description/config/kinematics/default/xarm6_default_kinematics.yaml` | True | False | True | 0.49 | `9e02c7ec66a06574d27cddebacfb294b8259b4e8b43cfc05d3bf38e8f358eac1` |
| `third_party/xarm_ros2/xarm_description/config/link_inertial/xarm6_type6_HT_BR2.yaml` | True | False | True | 1.19 | `ee278042008ad01e0fbc95a3ce0c270c898fe73bb46793c8a54daf34234c0f5c` |
| `third_party/xarm_ros2/xarm_description/meshes/xarm6/visual/link_base.stl` | True | False | True | 120.39 | `d3d61f2888bc39eecb8583babc7dddccb92f2ccd55aa32cd6f1dce6193feb3b6` |
| `third_party/xarm_ros2/xarm_description/meshes/xarm6/visual/link1.stl` | True | False | True | 183.68 | `103a81f9f00c217247f8b6c8f19b3afcbc9288852082389044d88a4a321cd8cd` |
| `third_party/xarm_ros2/xarm_description/meshes/xarm6/visual/link2.stl` | True | False | True | 1275.38 | `20ef55e34f8a63bb65108ed3134c4b22d911b55cd496ca5562b3978cba88ac77` |
| `third_party/xarm_ros2/xarm_description/meshes/xarm6/visual/link3.stl` | True | False | True | 207.50 | `f1503bc0c1fe8d39f4c509dba7288d9130954a35a1bcbaa0fca95d083fed3ffe` |
| `third_party/xarm_ros2/xarm_description/meshes/xarm6/visual/link4.stl` | True | False | True | 223.62 | `2a9952085fef91045380dac9825d9ad5a75867196933bc2cb1c3a635c23b0b7e` |
| `third_party/xarm_ros2/xarm_description/meshes/xarm6/visual/link5.stl` | True | False | True | 374.06 | `bfca01aae4255803ba2f5391c93836b91e1761e069275e38a9656ff2d53f45a4` |
| `third_party/xarm_ros2/xarm_description/meshes/xarm6/visual/link6.stl` | True | False | True | 126.84 | `b90cb1b373e65c035d7965bde92f4e2aedacdbbd57785a9018cd996b6b0b688e` |

## Simulation Code

| Path | Tracked | Generated | Portable | Size (KiB) | SHA256 |
| --- | --- | --- | --- | ---: | --- |
| `sim_mujoco/__init__.py` | True | False | True | 0.06 | `5c41e36a5ff7be345831ef345d2b55210dff5574e89fdf78d328bedbaa344acf` |
| `sim_mujoco/paths.py` | True | False | True | 2.28 | `1920c1f002ccbe46a24ebff86e79ba2d68bdd08a75b0be4edbf73691ccde6fbb` |
| `sim_mujoco/environment.py` | True | False | True | 9.31 | `542096ca8816076f1a1cc80ee3b20166e30df393bf2523f9d2b44c9df97e5e0b` |
| `sim_mujoco/collision.py` | True | False | True | 3.87 | `b8de48c86fbe659e1752ada0f61e2008edc9ac2990e7e56ddba2f16b8993b286` |
| `sim_mujoco/gripper_mapping.py` | True | False | True | 1.94 | `e7f4accbf6c1bc184d043ccb4ef35c344bd755dbc4d6ede1122c4a0547aee177` |
| `sim_mujoco/joint_mapping.py` | True | False | True | 1.08 | `03e00125567f3fa45ba521d7ec8a1e189f3cfb67a5dd8bc17e6740f629c48d3e` |
| `sim_mujoco/task_scenes.py` | True | False | True | 18.17 | `6b19b43b38fe5d7cb7ed5ed792dfa061d536380527adf838eecf11d68a2f936e` |
| `sim_mujoco/recording.py` | True | False | True | 0.60 | `6d6dca0cd8a9fbd61037bd48aa6c9ec4fb2172c8cabf6daa6006e14b20eda49b` |
| `sim_mujoco/remote_policy_observation.py` | True | False | True | 8.22 | `beb83c32ec3d5c9711e40c3ed0699a42cb223c50c9963e8984e8704f00f77157` |
| `sim_mujoco/remote_policy_control.py` | True | False | True | 5.32 | `4559b7d2344eccb5549c67353cd47bbbb41055d98167aad2f97b7b8bcffd3347` |
| `sim_mujoco/remote_policy_evaluation.py` | True | False | True | 11.94 | `f8f50d2110a7a6773a88137a7534aa486f86652f49794612ba021b75061480a3` |
| `sim_mujoco/scripts/generate_xarm6_mjcf.py` | True | False | True | 8.04 | `fcefc1838485c7a00beb3dde595484a34cd56c763aa3ab4b9ad060963afc4900` |
| `sim_mujoco/scripts/build_xarm6_pick_scene.py` | True | False | True | 20.72 | `74f01ca9c1c00212b031fee4003788774a5b1fd7e9891669a59686be41228507` |
| `sim_mujoco/scripts/camera_calibration_lib.py` | True | False | True | 26.14 | `aec896bbb35295610e9cbaefcbd6891a90c1d6f2578d418c8cef79cc174cdb73` |
| `sim_mujoco/scripts/render_task_scenes.py` | True | False | True | 4.57 | `4a4bac5e0c34f0813e6e52cabfbe4db120599fd2e23f46a613a201e2390905a7` |
| `sim_mujoco/scripts/smoke_test_headless_render.py` | True | False | True | 2.69 | `7b4eda32dd85dedd44280ed1586f638b70429ee2179e66568cd2d410aa0b4768` |
| `sim_mujoco/scripts/audit_kinematic_mapping.py` | True | False | True | 31.42 | `6e367ff9e03ade6ac3fe3bd8e64e392d08e5d191ae6b37eefc02a38ebaa38719` |

## Policy Integration

| Path | Tracked | Generated | Portable | Size (KiB) | SHA256 |
| --- | --- | --- | --- | ---: | --- |
| `policy_runtime/__init__.py` | True | False | True | 0.50 | `a40191438f4f7032a15bb04933b51c0b43739f333ce006c4f06ef4a9109ff877` |
| `policy_runtime/action_decoder.py` | True | False | True | 1.97 | `8c747c4bb93da5c8228bf46d252c6b5f087475fd9882bcdf654ced13d4e33220` |
| `policy_runtime/config.py` | True | False | True | 2.69 | `8108eed7e59982caf0e42367f10be4958296f6b3aa5a3baefdfe0748dbb01244` |
| `policy_runtime/environment_protocol.py` | True | False | True | 1.03 | `6470b073e66e0722295c8ea98b3c6305d5d0f37f4f37fbef4727167c68342630` |
| `policy_runtime/episode_logging.py` | True | False | True | 2.95 | `f0c434998a6d5c31a60bf83a31e2e56f268bdb7e4ced44a758ef83b654f87078` |
| `policy_runtime/evaluation.py` | True | False | True | 3.78 | `b7d44b2d30d6428cc866f29023e36cd3a29a82b5d433b2d72e6e84e05b4a29dc` |
| `policy_runtime/image_preprocessing.py` | True | False | True | 3.48 | `03e76ab407754bf9d76cd033ee2031c0de30dae51f0aa5ad3f4af66fd23c5bb2` |
| `policy_runtime/observation_builder.py` | True | False | True | 3.20 | `81789eaff2049bc4708b9268bd186516d36d552a81023bbe362f18801c441b46` |
| `policy_runtime/recording.py` | True | False | True | 4.11 | `cee351e00bfaabb3dc3c436d30fe92791d8db9dd62401fe36c24307375bbe59e` |
| `policy_runtime/remote_policy_client.py` | True | False | False | 4.55 | `0b09044add107e509cf552f7a1626ccaea7bebf422c11c8de1dcc2cc2d35ffbe` |
| `policy_runtime/runners.py` | True | False | True | 9.29 | `0be8eafeb9aa65f8441132aa525051176201b7dee7d415007a393146fad393b5` |
| `policy_runtime/safety.py` | True | False | True | 6.54 | `35b6d16b89c61ad7dafca54fc536d5ed60ef178a91d3be79bcc071bcab73acba` |
| `policy_runtime/schemas.py` | True | False | True | 1.97 | `45e5b81170cd4345e433fc5046e51dd01a73fa182de1641c9a90a9f2f1251780` |
| `sim_mujoco/scripts/run_remote_policy_dry_loop.py` | True | False | True | 6.07 | `363bd250d90eb799683d8988e992f52e837281c5e7d5f5ec31d8835417c54e10` |
| `sim_mujoco/scripts/run_remote_policy_closed_loop.py` | True | False | True | 23.02 | `d9531c4e473bc704510351009a5aa884a342b296361b5082b70fcd768bbd11f0` |
| `sim_mujoco/scripts/evaluate_remote_policy_interactive.py` | True | False | True | 13.58 | `ede4b512167277704bbaa2a08dba873b426b79b6355d07ae0994085a6d80de2c` |
| `sim_mujoco/scripts/test_remote_policy_mujoco.py` | True | False | True | 6.69 | `06dd5ac77c538e8d713bb0b280d980abe04a20f55cc76e175ef61dd0664cc8dc` |
| `sim_mujoco/scripts/test_remote_policy_once.py` | True | False | True | 1.52 | `5a303d12e079825460bb71bda8ce607bc4fb46cbaedd4ff2c644327deb6de6da` |

## Oracle Collection

| Path | Tracked | Generated | Portable | Size (KiB) | SHA256 |
| --- | --- | --- | --- | ---: | --- |
| `sim_mujoco/data_collection/__init__.py` | True | False | True | 0.07 | `529c1fa573df88aa0fc380563dd5f1e23a08401ac471bb6f7aa06e8a6964899e` |
| `sim_mujoco/data_collection/conversions.py` | True | False | True | 4.12 | `c64d0d30de74aeb940031dfe85769fa73a935cbe14943d2a869d00cf244148fd` |
| `sim_mujoco/data_collection/episode_recorder.py` | True | False | True | 20.33 | `dc554329117e95d0aa95f29f5652ada28bbace703afc28a043c9001f4d1490f9` |
| `sim_mujoco/data_collection/ik_solver.py` | True | False | True | 7.34 | `a4a606b58cd07ec666590653e65023b9376cf668f216ad35568e37f2643e4e7d` |
| `sim_mujoco/data_collection/lerobot_adapter.py` | True | False | True | 10.56 | `c53dde89df614273f26721f371c5881eeabecb754247f8944e73b7e2c8890a20` |
| `sim_mujoco/data_collection/oracle_controller.py` | True | False | True | 26.03 | `00115a1f6e123cc0efff89785b10d46e477d0834391d12476d4454a7ca41f671` |
| `sim_mujoco/data_collection/real_raw_recorder.py` | True | False | True | 10.03 | `5b10c2a86c8d5a76f2194d83ec318ad72b379e790addb48c75b7f1e2e2ed0f81` |
| `sim_mujoco/data_collection/task_success.py` | True | False | True | 1.04 | `fe2d08f43d24b36b19212897ddd152e884e2b2291a126795d7086a3b0ed7bca5` |
| `sim_mujoco/scripts/collect_oracle_data.py` | True | False | True | 14.68 | `0cfd0dc591c718c2b93e11dea11d1e4b52ec835f59fcdf6f2250421ee022effe` |
| `sim_mujoco/scripts/collect_real_raw_sim_data.py` | True | False | True | 16.25 | `badaaf3fa58f5c40fda275bb6e953a811bdb7aa85e3aeecd58e273a35a7bdb8c` |
| `sim_mujoco/scripts/test_scripted_oracle.py` | True | False | True | 5.86 | `c9501e4dc0550f0fa8b42656bfdd894cc2e37ff57a5fc41c6669fa796d44d9e1` |

## Dataset Tooling

| Path | Tracked | Generated | Portable | Size (KiB) | SHA256 |
| --- | --- | --- | --- | ---: | --- |
| `fine_tune/convert_xarm_raw_to_lerobot.py` | True | False | False | 18.94 | `b6bf646613205aef57ee874d6f588e3394884701dcdba3f93b6bc467bd531b48` |
| `fine_tune/xarm_lerobot_writer.py` | True | False | True | 9.29 | `b8b216269d6cd785a09b0779dea43deffc04edecb760a639df010354a43d9e89` |
| `fine_tune/smoke_test_openpi_xarm_dataset.py` | True | False | True | 8.96 | `3c673a6601bb0bcb836bbd36b96f9b7c98258ee2c36f238e92116f569d443301` |
| `sim_mujoco/scripts/convert_mujoco_to_lerobot.py` | True | False | True | 8.93 | `c9a9a89f766976ec9f788059d31902d2d3452f0cb5adbca63f5302642c44bf52` |
| `sim_mujoco/scripts/validate_mujoco_lerobot_dataset.py` | True | False | True | 22.05 | `97965ea85e37731ed24257af260ae4f32c7ece1727360aaf2375dca677d8059d` |
| `sim_mujoco/scripts/validate_real_raw_sim_data.py` | True | False | True | 7.44 | `b72ef518dc6823ca38f768d9f34f0b3644ab38e9507983a285eea4ba1b8b4e73` |
| `sim_mujoco/scripts/compare_real_sim_datasets.py` | True | False | True | 22.93 | `8252606697f4c8b19c9602d12d7dfcf988c1ab96019af905af6422a16ff5bd4a` |
| `sim_mujoco/scripts/prepare_mujoco_hf_ready.py` | True | False | False | 10.96 | `187021c596f3c52e4332be57dd12af1443e048edb31e12b7df43dba8308cf559` |
| `sim_mujoco/scripts/upload_mujoco_dataset_to_hf.py` | True | False | True | 10.74 | `f7719222e18de4dfefc3affc1254dec7be43360716d2a1062a22681192026ee6` |

## Configuration Documentation Tests

| Path | Tracked | Generated | Portable | Size (KiB) | SHA256 |
| --- | --- | --- | --- | ---: | --- |
| `.gitignore` | True | False | True | 1.30 | `a330587567fbd2c075cfd11ce04b2226e31a9dc4f035f0865b6ab26bcd81a2c3` |
| `.gitmodules` | True | False | True | 0.12 | `00d0c3024cd47dcfb8c9b2103c78dc822f053fbea9bfa4379ce9279c31f17f52` |
| `pytest.ini` | True | False | True | 0.21 | `f55773c6f29e02f9144d3f0741cb256e50af70d524f79a269a9e3096be3c780b` |
| `environment/mujoco_deltaai_requirements.txt` | True | False | True | 0.48 | `17fa963e887bef5feb03577ebfc2d92020721f0ce17d916307971d2ecad548c4` |
| `environment/mujoco_deltaai_environment.md` | True | False | True | 2.88 | `cbfdd740687a36a2b01adfe23b8e456bc64178c2d10ec13c8a9d81a6f7ec80b4` |
| `scripts/check_deltaai_mujoco_environment.py` | True | False | True | 7.82 | `072f1854902da699f30f2d2e2d8a067d9e425a86aa8361349839172212efae11` |
| `scripts/generate_mujoco_required_files_manifest.py` | True | False | True | 10.84 | `0481f156bd489876addff54af3b7aeeda96197c9d50dc180f4c244b4d81dea35` |
| `sim_mujoco/README.md` | True | False | False | 5.77 | `f0ecbde9b5e3c95132112d2fbcadfbbcd5e4535f4a1eb4b8cd92eb8f07700b05` |
| `sim_mujoco/DATA_COLLECTION.md` | True | False | False | 11.86 | `8feebeb97b1ec0f5d0684752ec2905054a9c3e16e74d9e334c771b090a3ac7b6` |
| `docs/mujoco_openpi_remote_inference_runbook.md` | True | False | False | 17.15 | `3233ff77b7bcf3fd8e790589fb7bef62c8d2477b483626fd3cf20399d4a73a35` |
| `docs/mujoco_task_scenes.md` | True | False | False | 5.10 | `0d23980a396a8deac335fd555c1120e99e40fda70f87a6f04408e6374d763ee7` |
| `tests/test_mujoco_chunk_execution.py` | True | False | True | 0.73 | `e746de2a897d2b0fd46e07a3ab2fcdbc4bbf076b38a80288c52bee87043cdf88` |
| `tests/test_mujoco_collisions.py` | True | False | True | 4.92 | `1732a64caaa9ac5f551b807aa94f50ab94fbf616d1353d43d0c7c9ac576fe444` |
| `tests/test_mujoco_data_conversions.py` | True | False | True | 2.77 | `f8dc84bae2751e521b6f258b98ab61e1c71202a180226d2f425304df0d6b14b5` |
| `tests/test_mujoco_episode_recorder.py` | True | False | True | 8.37 | `ce2f05278ccdfcc1f8ab28fb8be035d458dfc9bc5d7ff0a829dec96b20845f95` |
| `tests/test_mujoco_gripper_motion.py` | True | False | True | 1.14 | `95484d8ab3cee4086021e5425d612b0058b15d1981000f1a0d500b14aa9f55b3` |
| `tests/test_mujoco_hf_safety.py` | True | False | True | 4.52 | `0c9e462bc57f6767d8fdbf397efe813fb43dae3eddaaae3c53b109b16823f713` |
| `tests/test_mujoco_joint_mapping.py` | True | False | True | 2.92 | `25c8497155b49bfc08629a37cc8843b0bc8c29c04cc9c5db78f0e598c6c01840` |
| `tests/test_mujoco_lerobot_pipeline.py` | True | False | True | 11.12 | `5833a812381b805f6fda3a1c06a24015cd7c05d24c366ab866140bf56ec939d3` |
| `tests/test_mujoco_paths.py` | True | False | True | 1.33 | `e51a50ab936e5faf1bb4c52e4112ca238249fead1d800e1a10301ff10c943b17` |
| `tests/test_mujoco_scripted_oracle.py` | True | False | True | 4.15 | `93ea42c95afb25677ba391fe9c93076e39de4c0e3d0d28a6684ee46787c6b23d` |
| `tests/test_mujoco_task_scenes.py` | True | False | True | 6.35 | `0c97f6d69d3f13fa36ecf6388b1ef2aef938d5e93a341a7af42fdc5c49a13714` |
| `tests/test_policy_runtime_actions.py` | True | False | True | 1.30 | `dca28fdd8a8b232056beae2c0e6c4a837fb5ac6506c988e0bda3cf57d3394946` |
| `tests/test_policy_runtime_config.py` | True | False | False | 1.10 | `99cae926f92042c32a764a93f52c110e35ff2a9267fd1aa7a8e7a4fc69c71005` |
| `tests/test_policy_runtime_evaluation.py` | True | False | True | 1.65 | `8949239d1f4f574918afd132e57cd31d81433f1a57092fc69076b091ced55157` |
| `tests/test_policy_runtime_logging.py` | True | False | True | 0.91 | `552e552016168877f2bf59e6f5d8aaca03411dc4e89bda37696a02e7a925fbac` |
| `tests/test_policy_runtime_observation.py` | True | False | True | 2.48 | `c5ae64a9cd364be12dd2b9f5a204eb521b75a1547f63d2b24d314bb49fcb48ec` |
| `tests/test_policy_runtime_recording.py` | True | False | True | 0.74 | `2484c3ced56fcd402b0f05270c852cee0aeb62804a1cd852eccbf6af2149187e` |
| `tests/test_policy_runtime_safety.py` | True | False | True | 2.13 | `16d43accb71d4465fc8352f4f122a730539c5142d75c3cc22a0cbc35c2c9473d` |
| `tests/test_remote_policy_evaluation.py` | True | False | True | 5.58 | `6c3fc8e9c4c064901b77eaa614a00ad72b22f4b3c0813e0ca232af9707ad40d2` |
| `tests/test_remote_policy_pipeline.py` | True | False | True | 5.34 | `2b419ab359a97b48f4ef4b309e1f0e3ca0ea440485ce14f8ad4f782f07519819` |

Generated XML files must be regenerated from their source scripts
after source/config changes. Generated datasets, videos, calibration
imagery, caches, and checkpoints are intentionally absent.

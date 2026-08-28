# Menagerie UFACTORY xArm Gripper source

The gripper mechanics generated into `xarm6_pick_scene.xml` are ported from
Google DeepMind MuJoCo Menagerie `ufactory_xarm7/hand.xml`, pinned for this
integration at commit:

`da76818e269b82289eba39808e2fb91d679d6994`

Upstream source:

`https://github.com/google-deepmind/mujoco_menagerie/tree/da76818e269b82289eba39808e2fb91d679d6994/ufactory_xarm7`

The seven STL files are not duplicated. The builder reuses the byte-identical
UFACTORY meshes already vendored under:

`third_party/xarm_ros2/xarm_description/meshes/gripper/xarm/`

The local and pinned-upstream SHA-256 values were verified on 2026-08-14:

| file | SHA-256 |
|---|---|
| `base_link.stl` | `cdaa4cff22f7c9cff05c6a8ed32f94fd2b11a69d37dd97a159ccee2d8dd32f13` |
| `left_outer_knuckle.stl` | `501665812b08d67e764390db781e839adc6896a9540301d60adf606f57648921` |
| `left_finger.stl` | `a44756eb72f9c214cb37e61dc209cd7073fdff3e4271a7423476ef6fd090d2d4` |
| `left_inner_knuckle.stl` | `e8e48692ad26837bb3d6a97582c89784d09948fc09bfe4e5a59017859ff04dac` |
| `right_outer_knuckle.stl` | `75ca1107d0a42a0f03802a9a49cab48419b31851ee8935f8f1ca06be1c1c91e8` |
| `right_finger.stl` | `c5dee87c7f37baf554b8456ebfe0b3e8ed0b22b8938bd1add6505c2ad6d32c7d` |
| `right_inner_knuckle.stl` | `b41dd2c2c550281bf78d7cc6fa117b14786700e5c453560a0cb5fd6dfa0ffb3e` |

The UFACTORY model is BSD-3-Clause licensed. Its copyright and license are
preserved in `third_party/xarm_ros2/LICENSE`. Menagerie's package README also
identifies the model as BSD-3-Clause and UFACTORY-derived.

Integration adaptation: the hand is identity-attached to the existing xArm6
`link6` frame, matching this repository's xArm6/UFACTORY frame and preserving
the calibrated TCP and wrist-camera frame. The xArm7 arm is not imported.

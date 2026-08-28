# xArm training data contract

This document records the existing real/simulation training boundary. Phase 3
does not change its numeric meaning.

## Model-facing frame fields

| Stored LeRobot key | OpenPI meaning | Shape / type | Exact meaning |
| --- | --- | --- | --- |
| `image` | `observation/image` | RGB `uint8[480,640,3]` | Base camera image |
| `wrist_image` | `observation/wrist_image` | RGB `uint8[480,640,3]` | Wrist camera image |
| `state` | `observation/state` | `float32[7]` | Six xArm joint angles in radians, then hardware gripper raw |
| `actions` | `action` | `float32[7]` | Absolute target in the same ordering and units as `state` |
| `task` | language task | non-empty string | Training instruction/prompt |

The seventh CSV column is historically named `gripper_mm`, but its numeric
meaning at the training boundary is the xArm controller raw convention. It is
not an aperture measurement in millimetres. MuJoCo actuator radians and legacy
slide metres exist only behind `data.sim.generation.state_conversion`.

The gripper unit boundaries are:

- real CSV, canonical state/action, oracle inputs: xArm hardware raw units;
- canonical MuJoCo driver joints and actuator control: radians;
- legacy split-slide diagnostic models only: metres per finger slide;
- analytic/measured fingertip aperture: metres, used for geometry diagnostics
  and never written into the 7D training vector.

For the real raw format and the real-compatible simulation raw format, raw row
`t` is the observation and raw row `t + 1` supplies the absolute action. The
final raw row is intentionally dropped. LeRobot derives its own frame index,
episode index, and `frame_index / fps` timestamp while serializing.

## Non-model fields

A canonical in-memory frame additionally carries `episode_index`,
`frame_index`, `timestamp`, `source` (`real` or `sim`), and backend metadata.
These are provenance/indexing fields and are not added to model inputs. An
episode contains one backend's contiguous frames plus shared or backend-only
metadata.

Metadata classes are:

- shared: source identity, task identity, episode/frame indices, timestamps;
- simulation-only: seed/retry, randomization, oracle transitions, stability and
  acceptance results, scene variant, generation validation;
- real-only: collector timestamps/settings, camera serials, raw episode ID;
- excluded: formal policy-evaluation outcomes and review labels.

## Backend convergence

`data.sim.generation` and `data.real.collection` independently acquire or parse
backend data. Both produce the five identical model-facing fields above and use
`data.common.lerobot_writer` as the sole LeRobot writer. Backend acquisition
never enters `data.common`.

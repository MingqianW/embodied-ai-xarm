# Dataset schema

## Raw layout

```text
RAW_OUTPUT/
  OVERWRITE_MARKER.json
  run_config.json
  collection_manifest.json
  collection_summary.json
  accepted/<task_id>/episode_NNN/
    meta.json
    robot_log.csv
    gripper_events.csv
    realsense_0/*.png
    realsense_1/*.png
    realsense_2/*.png
    visual_manifest.json          # representative episodes
    contact_sheet.png             # representative episodes
    key_frames/*.png              # representative episodes
  failed_attempts/<task_id>/episode_NNN_attempt_NN/
  .staging/                       # empty after a clean completion
```

Failed attempts retain seeds and failure reasons but never appear in
`collection_manifest.json.completed`, conversion, or accepted directories.
Atomic JSON writes use a same-directory temporary file followed by replacement.

Each episode identifies `task_id`, `task_prompt`, requested and global indices,
base/retry/resolved seed, `scene_variant=clean`, initial conditions, transitions,
oracle plan, terminal reason, and validation metrics. Place initial-validation
metadata records ten steps and `initialization_frames_recorded=0`.

## State, action, and images

State and action are float32 vectors of length seven:

```text
[j1_rad, j2_rad, j3_rad, j4_rad, j5_rad, j6_rad, gripper_raw]
```

Raw row `t` is observation `t`; conversion uses the next robot-log row as the
absolute action target. Therefore an episode with N raw rows produces N−1
training frames. Data is 10 Hz. `realsense_0` is base, `realsense_1` is wrist,
and `realsense_2` is overview. Native images are RGB uint8 640×480.

## Manifests

`run_config.json` records absolute roots, camera/task config paths, Git SHA,
randomization, retry and verification values, plan, and overwrite inventory.
The continuously updated manifest separates completed episodes and failed
attempts. `collection_summary.json` becomes `complete: true` only when exact
task counts, zero distractors, validation metadata, and mandatory totals pass.

## Converted layout

The converter uses `data/common/lerobot_writer.py`, producing canonical
LeRobot v2.1 metadata (`meta/info.json`, `meta/tasks.jsonl`,
`meta/episodes.jsonl`), per-episode Parquet shards with embedded base/wrist RGB,
statistics assets, and `meta/mujoco_multitask_metadata.json`. Task indices follow
the YAML task order deterministically. The overview stream and failed attempts
are not training features.

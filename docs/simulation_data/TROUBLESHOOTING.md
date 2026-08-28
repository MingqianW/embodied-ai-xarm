# Troubleshooting

## Only one task or mixed prompts

Confirm the command uses the versioned YAML and the new module CLI, not the
legacy one-task or hard-coded multi-task script. Run `inspect`; it must list six
IDs and 200 episodes. `tasks.jsonl` must contain the six natural-language
prompts. Historical underscores are accepted only as input aliases.

## Pick reaches 5 cm then slips

Inspect `stable_grasp_failure_reason` and the verification metrics in
`meta.json`. A brief lift cannot complete the controller. Look at minimum/final
lift, maximum/final TCP-relative downward slip, grasp-region delta, table
contact, and robust final velocity. Do not loosen thresholds to improve yield;
change geometry-specific values only with physical justification, tests, smoke
evidence, and config/audit documentation.

The v3 pepper keeps the visible four-lobe silhouette and uses one convex torso
contact geom on the same free body. Its configured raw close target is `250`
and its grasp TCP offset is `-0.022 m`; a 20-seed retry-sequence sweep produced
13 strict full-window passes. Lower-yield trials remain failed attempts and are
never converted. Do not replace this with a weld or weaken the 1 cm slip gate.

The 26 mm `red_block`/`small_block` geometry uses raw close target `200` at the
standard `-0.011 m` TCP offset; that least-aggressive tested correction passed
5/5 strict retry-sequence trials. The larger blue/large blocks retain `211`.

## Place pepper is unstable at reset

Inspect the ten initial-validation samples and failure reason. Confirm the same
free `red_pepper` is initialized from the configured TCP transform after arm
noise, the gripper target is held, the pepper is above the table and outside the
ring, and drift is at most 5 mm. Do not restore `held_red_pepper`, a weld,
mocap lock, gravity changes, or per-step teleporting.

The v3 canonical reset uses a `0.012 m` simulated gripper half-width and a
TCP-to-pepper translation of `[0, 0, -0.030] m`. The convex contact profile is
enabled only for this Place task while the four visible lobes stay unchanged.
The canonical 20-seed diagnostic passed 20/20 seeds with maximum relative drift
between 0.55 and 0.84 mm and at least four gripper contacts per sample. Re-run
`place_grasp_sweep.sbatch` after any geometry, actuator, or reset change.

## Place body identity mismatch

The scene must report `target_body=object_identity=red_pepper`; active bodies are
only `red_pepper` and `ring`. Release changes the gripper state and metadata, not
the object body or pose. `initialization_frames_recorded` must remain zero.

## EGL or Python failures

Rendering belongs in Slurm with `MUJOCO_GL=egl` and
`PYOPENGL_PLATFORM=egl`. Verify `nvidia-smi` on the allocated compute node. Use
`/u/mw89/repos/openpi/.venv/bin/python` (Python 3.11), never the login-node
Python 3.6.

## Slurm timeout or OOM

Use `sacct -j JOB_ID --format=JobID,JobName%32,State,Elapsed,ExitCode,MaxRSS` and
inspect both Slurm logs. Do not run the phase interactively. Adjust a future job
request only from measured use; never silently restart into an existing output.

## Incomplete manifest or conversion missing tasks

Read `CODEX_STATUS.md`, `run_config.json`, `collection_manifest.json`, and the
phase status JSON. `complete: false` is intentional until all exact invariants
pass. Conversion refuses incomplete raw data and failed attempts. Compare task
counts and prompt audit CSVs before retrying.

## Permissions or ACL failure

Check `namei -l` on the exact root and `getfacl` on raw/converted roots. Re-run
the CLI `permissions` subcommand only for the four authorized roots. It never
accepts parents or siblings.

## Safe overwrite and interrupted replacement

Replacement requires `--overwrite`, exact canonical non-symlink paths, and a
pre-overwrite inventory under `/work/nvme/bfmk/mw89/logs`. If interrupted after
replacement, inspect `OVERWRITE_MARKER.json`, the outside inventory, status JSON,
and manifest before taking action. Never use a broad delete, clean, reset,
checkout, or implicit output root. Use `--resume` only when the saved run config
matches exactly; otherwise begin the explicitly authorized exact-root phase
again.

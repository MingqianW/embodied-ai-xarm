# Local MuJoCo Migration Validation

Date: 2026-07-29
Interpreter: `D:/miniconda/envs/mujoco-pi/python.exe` (Python 3.11.15)
Platform: Windows AMD64

## Summary

- Passed: syntax/import checks, active-scene regeneration and compile,
  environment checker, two-camera render, 85 unit tests, fixed scripted
  oracle, raw recorder, LeRobot conversion, validate-only conversion,
  full dataset validation, and Hugging Face preparation/upload dry run.
- Failed then resolved: the first dataset validator invocation could not
  write `C:/Users/26932/.cache/huggingface/datasets`. Setting `HF_HOME`,
  `HF_DATASETS_CACHE`, and `HF_HUB_CACHE` to an ignored writable output path
  made the same validation pass.
- Skipped: OpenPI transformed-batch validation and all live remote-policy
  inference, because migration checks must not require an OpenPI server.
- DeltaAI-gated: Linux aarch64, EGL, NVIDIA driver visibility, Slurm, and
  same-node π0.5 communication.

## Results

| Check | Status | Result |
| --- | --- | --- |
| Modified Python `py_compile` | PASSED | All migration and MuJoCo scripts compiled |
| Active scene regeneration | PASSED | Hash unchanged: `9f6d8ef3dc3be7ed8996dfc8513742e6257322efaafe6268536b639f4e80edd4` |
| Scene compile | PASSED | `nq=43`, `nv=38`, `nu=7`, `ncam=3`, timestep `0.002` |
| Environment checker | PASSED | Ready locally; expected warnings for AMD64 and unset EGL |
| Camera smoke test | PASSED | Base and wrist: `(480,640,3)`, RGB `uint8`, finite |
| MuJoCo unit tests | PASSED | 43 |
| Policy-runtime unit tests | PASSED | 20 |
| Remote-policy pipeline/evaluation unit tests | PASSED | 20 |
| Portable-path unit tests | PASSED | 2 |
| Fixed scripted oracle | PASSED | 1/1, stage `COMPLETE`, 72 actions |
| Historical fixed oracle gate | PASSED | 10/10 local-only result |
| Historical randomized oracle gate | PASSED | 20/20 at ±0.01 m, ±5°, 0.005 rad noise |
| Raw episode recording | PASSED | 1 successful episode, 72 samples |
| LeRobot conversion | PASSED | v2.1, 1 episode, 72 frames |
| Converter validate-only | PASSED | 1 existing episode; no pending frames |
| Dataset schema/content/loader | PASSED | Exact writer schema, RGB pixels, finite values |
| Action horizon | PASSED | `(10,7)` with boundary padding |
| OpenPI transformed batch | SKIPPED | Explicit `--skip-openpi-batch` |
| HF-ready preparation | PASSED | 10 hashed source entries |
| Hugging Face dry run | PASSED | 11 files, 4,117,650 bytes, no upload |
| HF remote repository query | UNAVAILABLE LOCALLY | Socket access denied; dry-run local validation still passed |
| Live OpenPI policy evaluation | SKIPPED | No server required or contacted |
| Linux aarch64 EGL | GATED UNTIL DELTAAI | Must run with `MUJOCO_GL=egl` on allocated GPU node |

## Commands

```powershell
$python = 'D:\miniconda\envs\mujoco-pi\python.exe'

& $python -m unittest discover -s tests -p 'test_mujoco*.py' -v
& $python -m unittest discover -s tests -p 'test_policy_runtime*.py' -v
& $python -m unittest discover -s tests -p 'test_remote_policy*.py' -v

& $python scripts/check_deltaai_mujoco_environment.py
& $python sim_mujoco/scripts/smoke_test_headless_render.py --task red_block --seed 0
& $python sim_mujoco/scripts/build_xarm6_pick_scene.py
& $python sim_mujoco/scripts/test_scripted_oracle.py --task red_block --episodes 1 --seed-start 0 --object-xy-range 0 --object-yaw-range-deg 0 --joint-noise 0
```

The end-to-end migration-validation artifacts were written under the ignored
directory `sim_mujoco/output/migration_validation/`.

## Important validation details

The first strict dataset-validation run failed only at the Hugging Face
dataset cache:

```text
PermissionError: [WinError 5] Access is denied:
C:\Users\26932\.cache\huggingface\datasets
```

The successful retry used:

```powershell
$env:HF_HOME = "$PWD\sim_mujoco\output\migration_validation\hf_cache"
$env:HF_DATASETS_CACHE = "$env:HF_HOME\datasets"
$env:HF_HUB_CACHE = "$env:HF_HOME\hub"
```

This confirms why writable scratch/cache environment variables are required
on DeltaAI.

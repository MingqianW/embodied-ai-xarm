# Isaac Sim + OpenPI xArm Pipeline

This adapter runs the repository's canonical xArm policy pipeline in local
Isaac Sim. It shares observation formatting, OpenPI transport, action decoding,
safety validation, logging, recording, and evaluation code with MuJoCo through
`policy_runtime/`. Isaac Sim and ROS 2 are not required by the ordinary test
suite.

Run all commands from the repository root in PowerShell. First inspect the
machine without changing it:

```powershell
.\sim_isaac\scripts\check_isaac_installation.ps1 `
  -OutputPath .\sim_isaac\output\installation_report.json
```

Set the path to an installed Isaac Sim distribution:

```powershell
$env:ISAAC_SIM_PATH = "C:\path\to\isaac-sim"
$IsaacPython = Join-Path $env:ISAAC_SIM_PATH "python.bat"
```

Validate the vendored xArm source, then expand and import it:

```powershell
& $IsaacPython .\sim_isaac\scripts\prepare_xarm_asset.py --validate-only
& $IsaacPython .\sim_isaac\scripts\prepare_xarm_asset.py `
  --expand-xacro --import-usd --headless
& $IsaacPython .\sim_isaac\scripts\inspect_xarm_asset.py
```

Inspect the two cameras and the exact OpenPI observation:

```powershell
& $IsaacPython .\sim_isaac\scripts\inspect_cameras.py
& $IsaacPython .\sim_isaac\scripts\test_observation_pipeline.py
```

With the OpenPI WebSocket server/tunnel available:

```powershell
& $IsaacPython .\sim_isaac\scripts\run_policy_dry_loop.py `
  --host 127.0.0.1 --port 18000 --iterations 5 --camera-debug

& $IsaacPython .\sim_isaac\scripts\run_policy_closed_loop.py `
  --host 127.0.0.1 --port 18000 `
  --max-policy-steps 20 --execute-chunk-steps 1 --record

& $IsaacPython .\sim_isaac\scripts\run_interactive_evaluation.py `
  --episodes 5 --record
```

The six configured collection prompts and their exact episode counts are
catalogued in `sim_isaac/config/tasks.yaml`.  To generate the complete 200
episode dataset (scene-mix metadata is intentionally ignored), start the
OpenPI policy server and run:

```powershell
& 'D:\isaacsim\python.bat' .\sim_isaac\scripts\collect_dataset.py `
  --host 127.0.0.1 --port 18000 `
  --output-dir .\sim_isaac\output\dataset_200
```

Each episode stores `initial/`, `policy_step_*/`, and `final/` state and RGB
images, action arrays, episode metadata, and a recording.  Verify the exact
plan without contacting the policy server using `--dry-run`; it writes
`dataset_manifest.json` with a total of 200 episodes.

For local scripted demonstrations, no policy server is needed.  The oracle
collector generates bounded waypoint actions inside Isaac Sim and saves the
same state/image/action fields:

```powershell
& 'D:\isaacsim\python.bat' .\sim_isaac\scripts\collect_oracle_dataset.py `
  --output-dir .\sim_isaac\output\oracle_dataset
```

Use `--plan-only` to validate the six prompts and 200-episode count without
launching Isaac Sim, or `--task pick_up_red_block --episodes 1` for a smoke
episode.  This oracle path is the appropriate one for local demonstration
collection; the policy-server path is only for evaluating a learned policy.

After collection, run the offline quality gate:

```powershell
& 'D:\isaacsim\python.bat' .\sim_isaac\scripts\validate_oracle_dataset.py `
  --input-dir .\sim_isaac\output\oracle_dataset
```

The report checks episode count, state/action shape and finiteness, RGB image
files, required oracle stages, and Isaac safety diagnostics.  It writes
`quality_report.json` and exits nonzero if any episode is incomplete or unsafe.
It also reports the simulated lift/placement metric; add
`--require-task-success` when only physically successful demonstrations should
be accepted.

Offline tests use the ordinary Python 3.11 environment:

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s sim_isaac/tests -v
```

Explicit Isaac tests use its launcher:

```powershell
$env:RUN_ISAAC_TESTS = "1"
& $IsaacPython -m unittest sim_isaac.tests.test_isaac_runtime -v

$env:RUN_ISAAC_POLICY_TESTS = "1"
& $IsaacPython -m unittest sim_isaac.tests.test_policy_integration -v
```

Generated USD, logs, videos, and camera comparisons are ignored under
`sim_isaac/generated/` and `sim_isaac/output/`. See
[`docs/ISAAC_SIM_SETUP.md`](../docs/ISAAC_SIM_SETUP.md) for installation,
calibration, safety, tests, troubleshooting, and known limitations.

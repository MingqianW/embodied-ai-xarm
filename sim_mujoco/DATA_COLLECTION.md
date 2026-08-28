# MuJoCo data collection

The current reusable, clean-scene, six-task pipeline is documented in
[`docs/simulation_data/README.md`](../docs/simulation_data/README.md).

Use the versioned config at
`configs/data/sim/generation/clean_multitask_stable_v3.yaml` and the
stable `python -m data.sim.generation.cli` entry point. Heavy tests,
rendering, collection, conversion, and full decoding run only through the
self-contained scripts in `slurm/simulation_data/`.

The older `collect_oracle_data.py` and `collect_real_raw_sim_data.py` scripts
remain for historical reproducibility. They are not the v3 workflow and their
hard-coded prompt/distractor behavior must not be used to regenerate v3.

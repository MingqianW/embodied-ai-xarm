# FarmShare deployment templates

This directory contains site environment and scheduler-resource templates only.
It reuses `cluster.cli` and `cluster/jobs/run_workflow.sbatch`; it does not own
model loading, inference, simulation, tasks, scoring, or a second Slurm tree.

Read the [operating guide](../../docs/commands/farmshare_sim_evaluation.md)
before submitting jobs. Confirm coursework/unsponsored-research eligibility,
then verify the account, partitions, QoS, and limits on the actual cluster.

`environment.example.sh` requires operator-provided deployment paths and an
account obtained from `sacctmgr`. Keep those machine-specific values outside
the checkout. The resource JSON overrides existing workflows only. Its
`environment-check` request is 8 CPU/32G/1 GPU/30 minutes; its evaluation
request is 16 CPU/64G/1 GPU/12 hours. These are initial requests compatible
with the observed 4G-per-CPU GPU partition limit, not measured model sizing.
Measure inference/rendering memory and rollout time before formal evaluation.

The existing `formal-sim-evaluation` workflow remains a client of a separately
verified server. These templates do not yet launch or validate the target
policy service. Server/client deployment on the same allocated compute node
must be verified before using localhost.

Absent `XARM_SLURM_QOS` and `XARM_SLURM_RESOURCE_CONFIG`, the existing DeltaAI
submission behavior and resource requests remain unchanged.

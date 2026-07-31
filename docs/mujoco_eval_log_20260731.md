# Success Rate Table

| Task | Model A | Model B | Model C |
|---|---:|---:|---:|
| Red pepper | 4/20 (20.0%) | 0/20 (0.0%) | 4/20 (20.0%) |
| Blue block | 0/20 (0.0%) | 0/20 (0.0%) | 7/20 (35.0%) |
| Red block | 20/20 (100.0%) | 20/20 (100.0%) | 20/20 (100.0%) |
| Smallest block | 8/20 (40.0%) | 4/20 (20.0%) | 8/20 (40.0%) |
| Largest block | 1/20 (5.0%) | 0/20 (0.0%) | 11/20 (55.0%) |
| Place red pepper in ring | 20/20 (100.0%) | 0/20 (0.0%) | 0/20 (0.0%) |
| Overall | 53/120 (44.2%) | 24/120 (20.0%) | 50/120 (41.7%) |

- Per-task denominator: 20 episodes
- Overall denominator: 120 episodes per model
- Control setting: 5 actions per inference, maximum 50 inferences
  
| Model | Initialization | Training Data | Data Ratio | Normalization | Training Method |
|---|---|---|---|---|---|
| A | π0.5 base model | Real + Sim | 1:1 | Joint normalization | Full-parameter fine-tuning for approximately 30k steps |
| B | π0.5 base model | Sim only | 100% Sim | Simulation-specific normalization | Full-parameter fine-tuning for approximately 30k steps |
| C | Real-only xArm checkpoint trained for 30k steps | Real + Sim | 1:1 | Joint normalization | Continuation full-parameter fine-tuning for approximately 30k steps |
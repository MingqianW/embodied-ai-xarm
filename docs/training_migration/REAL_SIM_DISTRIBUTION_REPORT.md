# Real/Simulation Distribution Gate

**Result: PASS**

- Real: 198 episodes, 22618 frames
- Simulation: 198 episodes, 14425 frames
- Canonical schema, units, ordering, temporal alignment, prompt, and RGB camera contract passed.
- Normalization remains the checkpoint-30000 real-data q01/q99 asset.
- Simulation state overall q01/q99 outside rate: 0.235187
- Simulation transformed-action overall q01/q99 outside rate: 0.187135
- Numeric and visual/physical domain shift is reported but is not a semantic failure.

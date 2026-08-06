# Tasks and prompts

| Task ID | Canonical emitted prompt | Episodes | Required objects |
|---|---|---:|---|
| `red_pepper` | `pick up the red pepper` | 50 | red pepper |
| `blue_block` | `pick up the blue block` | 25 | blue block |
| `red_block` | `pick up the red block` | 25 | red block |
| `smallest_block` | `pick up the smallest block` | 25 | small and large blocks |
| `largest_block` | `pick up the largest block` | 25 | small and large blocks |
| `place_red_pepper_in_ring` | `place the red pepper in the ring` | 50 | free red pepper and ring |

Machine IDs and natural-language prompts are separate fields. Every raw episode
stores `task_id` and `task_prompt`; converted tasks use only the canonical prompt.
Prompts never vary by seed, retry, or scene variant.

## Input-only aliases

| Alias examples | Resolves to |
|---|---|
| `pick_up_the_red_pepper`, `red pepper` | `red_pepper` |
| `pick_up_the_blue_block`, `pick up the light blue block`, `blue block` | `blue_block` |
| `pick_up_the_red_block`, `red block` | `red_block` |
| `pick_up_the_smallest_block`, `smallest block`, `smallest` | `smallest_block` |
| `pick_up_the_largest_block`, `largest block`, `largest` | `largest_block` |
| `place_the_red_pepper_in_the_ring` | `place_red_pepper_in_ring` |

Historical real data includes underscore prompts and, in later datasets,
natural-language prompts. Do not mutate those datasets. For joint real/simulation
training, normalize task text through
`sim_mujoco.data_generation.registry.resolve_task_id` and then
`canonical_prompt`. Tests assert that historical aliases and simulation IDs
resolve to the same string.

`smallest_block` and `largest_block` are relational tasks, so both comparison
blocks are required even in a clean scene. They are not distractors. Place starts
with the pepper held in the air and contains no tabletop Pick behavior.

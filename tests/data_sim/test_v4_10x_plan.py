from pathlib import Path

from data.sim.generation.collection import resolve_seed
from data.sim.generation.config import load_pipeline_config
from data.sim.generation.plans import expected_roots


V3 = Path(
    "configs/data/sim/generation/"
    "clean_multitask_stable_v3.yaml"
)

V4 = Path(
    "configs/data/sim/generation/"
    "clean_multitask_stable_v4_10x_real.yaml"
)

REAL_COUNTS = {'red_pepper': 50, 'blue_block': 24, 'red_block': 25, 'smallest_block': 24, 'largest_block': 25, 'place_red_pepper_in_ring': 50}

EXPECTED_SIM_COUNTS = {
    task_id: count * 10
    for task_id, count in REAL_COUNTS.items()
}


def test_v4_is_strictly_ten_times_actual_real_data() -> None:
    v3 = load_pipeline_config(V3)
    v4 = load_pipeline_config(V4)

    actual = {
        task.task_id: task.episodes
        for task in v4.tasks
    }

    # v3 remains the independent 200-episode simulation reference.
    assert v3.total_episodes == 200

    # v4 is defined from the actual 198-episode real dataset.
    assert sum(REAL_COUNTS.values()) == 198
    assert actual == EXPECTED_SIM_COUNTS
    assert v4.total_episodes == 1980

    assert all(
        task.distractor_episodes == 0
        for task in v4.tasks
    )
    assert v4.distractor_count == 0

    assert set(vars(v4.outputs).values()) == expected_roots(
        v4.dataset_version
    )


def test_v4_seed_ranges_do_not_overlap_v3_or_each_other() -> None:
    v3 = load_pipeline_config(V3)
    v4 = load_pipeline_config(V4)

    v3_seeds = {
        resolve_seed(
            task,
            episode,
            retry,
            v3.seed_retry_stride,
        )
        for task in v3.tasks
        for episode in range(task.episodes)
        for retry in range(v3.max_attempts_per_episode)
    }

    v4_seeds = set()
    task_ranges = []

    for task in v4.tasks:
        task_seeds = {
            resolve_seed(
                task,
                episode,
                retry,
                v4.seed_retry_stride,
            )
            for episode in range(task.episodes)
            for retry in range(v4.max_attempts_per_episode)
        }

        assert not task_seeds & v4_seeds
        v4_seeds |= task_seeds
        task_ranges.append(
            (min(task_seeds), max(task_seeds))
        )

    assert not v3_seeds & v4_seeds

    for index, first in enumerate(task_ranges):
        for second in task_ranges[index + 1:]:
            assert (
                first[1] < second[0]
                or second[1] < first[0]
            )

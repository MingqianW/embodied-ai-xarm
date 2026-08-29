from __future__ import annotations

from collections import Counter

from data.common.records import SourceBackend
from training.mixing.sampler import SampleRef, TrajectoryRef, sample_batches, trajectory_batches
from training.mixing.strategies import MixingStrategy


def _samples(real: int, sim: int) -> list[SampleRef]:
    return [
        SampleRef("real", SourceBackend.REAL, index, 0) for index in range(real)
    ] + [SampleRef("sim", SourceBackend.SIM, index, 0) for index in range(sim)]


def test_real_only_cycles_deterministically_at_exhaustion() -> None:
    strategy = MixingStrategy.single(SourceBackend.REAL, seed=7)
    first = sample_batches(_samples(3, 0), strategy, batch_size=2, num_batches=4)
    second = sample_batches(_samples(3, 0), strategy, batch_size=2, num_batches=4)
    assert first == second
    assert {item.source for batch in first for item in batch} == {SourceBackend.REAL}
    assert len({item.episode_index for batch in first for item in batch}) == 3


def test_fixed_one_to_one_is_exact_per_batch_despite_source_size_difference() -> None:
    batches = sample_batches(
        _samples(2, 11),
        MixingStrategy.per_batch(8, 8, seed=11),
        batch_size=16,
        num_batches=5,
    )
    for batch in batches:
        counts = Counter(item.source for item in batch)
        assert counts == {SourceBackend.REAL: 8, SourceBackend.SIM: 8}


def test_fixed_one_to_ten_is_a_stream_schedule_not_per_batch_composition() -> None:
    batches = sample_batches(
        _samples(2, 3),
        MixingStrategy.sample_ratio(1, 10, seed=5),
        batch_size=16,
        num_batches=11,
    )
    stream = [item.source for batch in batches for item in batch]
    assert stream[:11] == [SourceBackend.REAL] + [SourceBackend.SIM] * 10
    assert Counter(stream) == {SourceBackend.REAL: 16, SourceBackend.SIM: 160}
    assert len({item.episode_index for item in sum((list(batch) for batch in batches), []) if item.source is SourceBackend.REAL}) == 2


def test_sim_only_has_no_real_starvation_concept() -> None:
    batches = sample_batches(
        _samples(0, 2),
        MixingStrategy.single(SourceBackend.SIM),
        batch_size=3,
        num_batches=2,
    )
    assert all(item.source is SourceBackend.SIM for batch in batches for item in batch)


def test_global_shuffle_preserves_frames_inside_each_trajectory() -> None:
    trajectories = []
    for source, dataset in ((SourceBackend.REAL, "real"), (SourceBackend.SIM, "sim")):
        for episode in range(2):
            frames = tuple(SampleRef(dataset, source, episode, frame) for frame in range(3))
            trajectories.append(TrajectoryRef(dataset, source, episode, frames))
    strategy = MixingStrategy.trajectory_shuffle(seed=19)
    batches = trajectory_batches(trajectories, strategy, batch_size=4, num_batches=3)
    stream = [item for batch in batches for item in batch]
    assert stream == [item for batch in trajectory_batches(trajectories, strategy, batch_size=4, num_batches=3) for item in batch]
    for offset in range(0, len(stream), 3):
        chunk = stream[offset : offset + 3]
        assert [item.frame_index for item in chunk] == [0, 1, 2]
    assert {item.source for item in stream} == {SourceBackend.REAL, SourceBackend.SIM}

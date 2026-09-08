from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from data.common.records import SourceBackend
from training.mixing.sampler import DeterministicSourceStream, SampleRef
from training.mixing.strategies import MixingStrategy
from training.configs.experiments import get_experiment
from training.openpi.mixed_loader import (
    OpenPITrainingRuntime,
    SamplingCounters,
    DatasetFrameRef,
    _LoadedDataset,
    _normalization_manifest,
    format_mixed_training_summary,
)


def _pools(real_count: int, sim_count: int) -> dict[SourceBackend, tuple[SampleRef, ...]]:
    return {
        SourceBackend.REAL: tuple(
            SampleRef("real", SourceBackend.REAL, index, 0)
            for index in range(real_count)
        ),
        SourceBackend.SIM: tuple(
            SampleRef("sim", SourceBackend.SIM, index, 0)
            for index in range(sim_count)
        ),
    }


def _stream_sources(
    real_count: int, sim_count: int, strategy: MixingStrategy, *, samples: int = 22_000
) -> tuple[list[SourceBackend], DeterministicSourceStream[SampleRef]]:
    stream = DeterministicSourceStream(_pools(real_count, sim_count), strategy, batch_size=20)
    return [stream.item_at(index).source for index in range(samples)], stream


def test_equal_sampling_is_not_derived_from_a_one_to_ten_dataset_size_ratio() -> None:
    strategy = MixingStrategy.real_sim_weighted_sampling(
        real_sampling_weight=1.0, sim_sampling_weight=1.0, seed=5
    )
    sources, stream = _stream_sources(100, 1_000, strategy)
    counts = Counter(sources)
    assert counts[SourceBackend.REAL] / len(sources) == pytest.approx(0.5, abs=0.02)
    assert stream.source_pool_sizes == {SourceBackend.REAL: 100, SourceBackend.SIM: 1_000}


def test_one_to_ten_sampling_keeps_all_real_and_simulation_examples_available() -> None:
    strategy = MixingStrategy.real_sim_weighted_sampling(
        real_sampling_weight=1.0, sim_sampling_weight=10.0, seed=7
    )
    sources, stream = _stream_sources(100, 1_000, strategy)
    counts = Counter(sources)
    assert counts[SourceBackend.REAL] / len(sources) == pytest.approx(1 / 11, abs=0.015)
    assert counts[SourceBackend.SIM] / len(sources) == pytest.approx(10 / 11, abs=0.015)
    assert stream.source_pool_sizes == {SourceBackend.REAL: 100, SourceBackend.SIM: 1_000}


def test_increasing_simulation_dataset_size_does_not_change_weighted_domain_selection() -> None:
    strategy = MixingStrategy.real_sim_weighted_sampling(
        real_sampling_weight=1.0, sim_sampling_weight=10.0, seed=11
    )
    sources_1k, stream_1k = _stream_sources(100, 1_000, strategy)
    sources_5k, stream_5k = _stream_sources(100, 5_000, strategy)
    assert sources_1k == sources_5k
    assert stream_1k.source_pool_sizes[SourceBackend.SIM] == 1_000
    assert stream_5k.source_pool_sizes[SourceBackend.SIM] == 5_000


def test_changing_sampling_weights_does_not_change_physical_dataset_sizes() -> None:
    equal = MixingStrategy.real_sim_weighted_sampling(
        real_sampling_weight=1.0, sim_sampling_weight=1.0, seed=13
    )
    one_to_ten = MixingStrategy.real_sim_weighted_sampling(
        real_sampling_weight=1.0, sim_sampling_weight=10.0, seed=13
    )
    equal_sources, equal_stream = _stream_sources(100, 1_000, equal)
    weighted_sources, weighted_stream = _stream_sources(100, 1_000, one_to_ten)
    assert equal_stream.source_pool_sizes == weighted_stream.source_pool_sizes == {
        SourceBackend.REAL: 100,
        SourceBackend.SIM: 1_000,
    }
    assert Counter(equal_sources)[SourceBackend.REAL] > Counter(weighted_sources)[SourceBackend.REAL]


def test_weighted_sampling_stores_each_physical_pool_once_and_normalizes_weights() -> None:
    ratio = MixingStrategy.real_sim_weighted_sampling(
        real_sampling_weight=1.0, sim_sampling_weight=10.0
    )
    probabilities = MixingStrategy.real_sim_weighted_sampling(
        real_sampling_weight=0.090909, sim_sampling_weight=0.909091
    )
    assert dict(ratio.sampling_probabilities)[SourceBackend.REAL] == pytest.approx(1 / 11)
    assert dict(probabilities.sampling_probabilities)[SourceBackend.REAL] == pytest.approx(1 / 11, abs=1e-5)
    _, stream = _stream_sources(100, 1_000, ratio, samples=1)
    assert sum(stream.source_pool_sizes.values()) == 1_100
    assert stream.source_pool_sizes[SourceBackend.REAL] == 100
    assert stream.source_pool_sizes[SourceBackend.SIM] == 1_000


def test_consumption_counters_report_observed_domain_fractions() -> None:
    counters = SamplingCounters((SourceBackend.REAL, SourceBackend.SIM))
    for _ in range(2):
        counters.record(SourceBackend.REAL)
    for _ in range(8):
        counters.record(SourceBackend.SIM)
    snapshot = counters.snapshot()
    assert snapshot["counts"] == {"real": 2, "sim": 8}
    assert snapshot["fractions"] == {"real": 0.2, "sim": 0.8}


def test_mixed_normalization_manifest_declares_physical_pool_weighting(tmp_path: Path) -> None:
    experiment = get_experiment("pi05_xarm_real1_sim10_stratified")
    runtime = OpenPITrainingRuntime(
        experiment,
        object(),
        {spec.dataset_id: tmp_path / spec.dataset_id for spec in experiment.datasets.datasets},
    )
    manifest = _normalization_manifest(runtime, "test_asset")
    assert manifest["weighting"] == {
        "mode": "uniform_selected_physical_frames",
        "replays_training_mixing": False,
        "description": (
            "Every frame in the selected DatasetSpec pool is included once; "
            "domain sampling weights do not change normalization statistics."
        ),
    }
    assert {row["dataset_id"] for row in manifest["pool"]} == {
        "real_xarm_pi05_20260703",
        "sim_mujoco_stable_v4_10x_real",
    }


def test_startup_summary_declares_sizes_sampling_and_step_based_semantics(tmp_path: Path) -> None:
    experiment = get_experiment("pi05_xarm_real1_sim10_stratified")
    runtime = OpenPITrainingRuntime(
        experiment,
        object(),
        {spec.dataset_id: tmp_path / spec.dataset_id for spec in experiment.datasets.datasets},
    )
    loaded = tuple(
        _LoadedDataset(
            spec,
            tmp_path / spec.dataset_id,
            object(),
            (DatasetFrameRef(spec.dataset_id, spec.source, 0, 0, 0),),
            ((DatasetFrameRef(spec.dataset_id, spec.source, 0, 0, 0),),),
        )
        for spec in experiment.datasets.datasets
    )
    summary = format_mixed_training_summary(
        runtime,
        loaded,
        SimpleNamespace(batch_size=16, num_train_steps=15_001),
    )
    assert "real dataset:\n  episodes: 1\n  samples: 1" in summary
    assert "sim dataset:\n  episodes: 1\n  samples: 1" in summary
    assert "legacy exact schedule: real=1, sim=10" in summary
    assert "every selected frame once (not training sampling weights)" in summary
    assert "epoch: virtual step-based stream; not derived from combined dataset length" in summary

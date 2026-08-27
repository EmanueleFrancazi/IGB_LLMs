"""The production payload builders, driven end to end without a model.

The risky part of a writer is not the NPZ call -- it is the **mapping**, where a
field can be wired to the wrong statistic and nothing complains. That mapping
used to live inside ``main()`` behind a tokenizer, a model and the whole
nucleus-label recovery, so no test could reach it.

Both writers now delegate to an extracted payload builder and an extracted NPZ
writer, and ``main()`` calls those and nothing else. So the boundary tested here
is the production one::

    independently computed analysis results
        -> production payload builder
        -> production NPZ writer
        -> np.load(allow_pickle=False) / authoritative loader

The inputs are **sentinels**: delta, within, between, the null, and the two
references all take distinct values, so a field copied from its neighbour cannot
pass. A symmetric fixture would let exactly the bug this exists to catch through.
"""

from __future__ import annotations

import importlib
import importlib.util

import numpy as np
import pytest

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _script(stem: str):
    """Load a script by path; ``scripts/`` is not an importable package."""

    path = REPO_ROOT / "scripts" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"_seam_{stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


nucleus_writer = _script("write_nucleus_clustering_artifact")
cross_writer = _script("write_cross_partition_artifact")

from llm_behavior_lab.analysis.nucleus_clustering_artifact import (  # noqa: E402
    ARTIFACT_NAME,
    load_nucleus_clustering_artifact,
)

D = 12
K = 5

#: Every statistic gets its own decade so a mis-wire is unmissable.
DELTA = [0.10, 0.11]
WITHIN = [0.30, 0.31]
BETWEEN = [0.20, 0.21]
NULL_MEAN = [0.001, 0.002]
REFERENCE = {"target": 0.41, "greedy": 0.52}


def _result(maps: int):
    """A loader-shaped sweep result with sentinel statistics."""

    temps = [0.6, 1.0]
    return {
        "sampling_temperatures": np.asarray(temps),
        "loss_temperatures": np.full(2, 1.0),
        "temperatures": tuple(temps),
        "by_temperature": [
            {
                "temperature": t, "sampling_temperature": t, "loss_temperature": 1.0,
                "population": {
                    "delta": DELTA[i], "within": WITHIN[i], "between": BETWEEN[i],
                    "num_classes": 3, "num_represented": 3, "num_positions": D,
                },
                "null": {
                    "delta_mean": NULL_MEAN[i], "delta_low": -0.05,
                    "delta_high": 0.05,
                },
                "support": {
                    "num_represented": 3, "num_qualifying": 3, "num_singletons": 0,
                    "singleton_fraction": 0.0, "positions_in_qualifying": D,
                    "fraction_positions_in_qualifying": 1.0, "largest_class": 4,
                    "median_class": 4, "num_within_pairs": 12,
                    "num_between_pairs": 96,
                },
                "display": {
                    "classes": np.asarray([3, 4, 5]),
                    "matrix": np.zeros((3, 3)),
                    "counts": np.asarray([4, 4, 4]),
                },
                "num_positions": D, "num_positions_excluded": 0,
                "sketch_dimension": K,
            }
            for i, t in enumerate(temps)
        ],
        "references": {
            g: {"population": {"delta": REFERENCE[g]},
                "null": {"delta_low": -0.04, "delta_high": 0.04}}
            for g in ("target", "greedy")
        },
        "references_by_loss_temperature": {
            1.0: {
                g: {"population": {"delta": REFERENCE[g]},
                    "null": {"delta_low": -0.04, "delta_high": 0.04}}
                for g in ("target", "greedy")
            }
        },
        "min_support": 2, "permutations": 8, "permutation_seed": 20240918,
    }


class _Record:
    """Only what the payload builder reads from a record."""

    def __init__(self, maps: int) -> None:
        self.sketch_map_count = maps


# -- nucleus: sentinel mapping ------------------------------------------------------


def test_each_nucleus_field_comes_from_its_own_statistic() -> None:
    """Sentinels, so a field copied from its neighbour cannot pass."""

    arrays = nucleus_writer._nucleus_artifact_arrays(
        _result(1), _Record(1), np.zeros((2, D), dtype=np.int64)
    )

    assert np.allclose(arrays["delta"], DELTA)
    assert np.allclose(arrays["within"], WITHIN)
    assert np.allclose(arrays["between"], BETWEEN)
    assert np.allclose(arrays["null_delta_mean"], NULL_MEAN)
    assert float(arrays["reference_target_delta"]) == REFERENCE["target"]
    assert float(arrays["reference_greedy_delta"]) == REFERENCE["greedy"]

    # And the per-map vectors track their own statistic, not a neighbour's.
    assert np.allclose(arrays["delta_per_map"][:, 0], DELTA)
    assert np.allclose(arrays["within_per_map"][:, 0], WITHIN)
    assert np.allclose(arrays["between_per_map"][:, 0], BETWEEN)
    assert np.allclose(arrays["null_delta_per_map_mean"][:, 0], NULL_MEAN)

    assert not np.allclose(arrays["delta_per_map"], arrays["within_per_map"])
    assert not np.allclose(arrays["delta_per_map"], arrays["between_per_map"])
    assert not np.allclose(arrays["within_per_map"], arrays["between_per_map"])
    assert float(arrays["reference_target_delta_per_map"][0]) != float(
        arrays["reference_greedy_delta_per_map"][0]
    )


def test_delta_per_map_is_within_minus_between_on_a_valid_fixture() -> None:
    """Checked on real b1 output, where the identity actually has to hold."""

    import importlib as _il

    gc = _il.import_module("llm_behavior_lab.analysis.gradient_clustering")
    shared = _il.import_module("test_figure_uncertainty_artists")

    per_map = gc.gradient_clustering_per_map(shared._record(4), permutations=8)

    assert np.allclose(
        per_map["delta_per_map"],
        per_map["within_per_map"] - per_map["between_per_map"],
        rtol=0, atol=1e-15,
    )


def test_the_production_writer_round_trips_through_the_loader(tmp_path) -> None:
    """Payload builder -> production NPZ writer -> authoritative loader."""

    arrays = nucleus_writer._nucleus_artifact_arrays(
        _result(1), _Record(1), np.zeros((2, D), dtype=np.int64)
    )
    path = tmp_path / ARTIFACT_NAME
    nucleus_writer._write_nucleus_artifact(path, arrays)

    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            assert data[key].dtype != object, key
        assert int(data["map_count"]) == 1
        assert int(data["degrees_of_freedom"]) == 0
        assert bool(data["uncertainty_available"]) is False
        assert np.isnan(data["delta_sd"])
        assert np.isnan(data["delta_se"])
        assert data["delta_per_map"].shape == (2, 1)
        assert data["delta_per_map"].dtype == np.float64

    loaded = load_nucleus_clustering_artifact(tmp_path, ARTIFACT_NAME)

    assert loaded["map_count"] == 1
    assert loaded["degrees_of_freedom"] == 0
    assert loaded["uncertainty_available"] is False
    # Historical point estimates survive unchanged.
    assert [e["population"]["delta"] for e in loaded["by_temperature"]] == DELTA
    assert [e["population"]["within"] for e in loaded["by_temperature"]] == WITHIN
    assert [e["population"]["between"] for e in loaded["by_temperature"]] == BETWEEN
    # And the additive values are readable.
    assert np.allclose(loaded["uncertainty"]["delta_per_map"][:, 0], DELTA)


def test_a_historical_artifact_lacking_the_new_fields_still_loads(tmp_path) -> None:
    """Defaults, not failure: such an artifact came from one map."""

    arrays = nucleus_writer._nucleus_artifact_arrays(
        _result(1), _Record(1), np.zeros((2, D), dtype=np.int64)
    )
    historical = {
        key: value
        for key, value in arrays.items()
        if not key.endswith(("_per_map", "_sd", "_se"))
        and key not in ("map_count", "degrees_of_freedom", "uncertainty_available")
    }
    path = tmp_path / ARTIFACT_NAME
    nucleus_writer._write_nucleus_artifact(path, historical)

    loaded = load_nucleus_clustering_artifact(tmp_path, ARTIFACT_NAME)

    assert loaded["map_count"] == 1
    assert loaded["degrees_of_freedom"] == 0
    assert loaded["uncertainty_available"] is False
    assert loaded["uncertainty"]["delta_se"] is None
    assert [e["population"]["delta"] for e in loaded["by_temperature"]] == DELTA


# -- cross-partition: sentinel mapping ------------------------------------------------


C_SAME, C_DIFF, DELTA_CROSS = 0.71, 0.32, 0.39
MIXTURE_PLUG_IN = 0.63


def _cross_inputs():
    classes = np.asarray([3, 4, 5])
    pooled = {
        "classes": classes, "c_same": C_SAME, "c_different": C_DIFF,
        "delta_cross": DELTA_CROSS, "num_same_pairs": 10,
        "num_different_pairs": 20, "num_true_positive_positions": 4,
    }
    null = {"delta_mean": 0.0, "delta_std": 0.02, "delta_low": -0.05,
            "delta_high": 0.05, "permutations": 8, "seed": 20240918}
    contingency = {
        "num_true_positive_positions": 4, "true_positive_fraction": 0.33,
        "num_target_classes": 3, "num_greedy_classes": 3, "num_cells": 6,
        "counts": np.zeros((3, 3), dtype=np.int64),
        "target_ids": classes, "greedy_ids": classes,
    }
    mixture = {
        "tokens": classes, "support": np.asarray([4, 4, 4]),
        "similarity": np.asarray([0.6, 0.63, 0.66]),
        "residual_norm": np.asarray([0.1, 0.1, 0.1]), "num_classes": 3,
        "median_similarity": MIXTURE_PLUG_IN, "iqr_similarity": 0.06,
        "support_weighted_similarity": 0.63,
        "metric": "projected-space cosine between sketch-space mean directions",
    }
    drawn = {"matrix": np.zeros((3, 3)), "contingency": np.zeros((3, 3))}
    return pooled, null, contingency, mixture, drawn, classes


def test_the_cross_partition_payload_keeps_each_statistic_distinct(tmp_path) -> None:
    pooled, null, contingency, mixture, drawn, classes = _cross_inputs()
    counts = np.asarray([4, 4, 4])

    arrays = cross_writer._cross_partition_arrays(
        _Record(1), pooled, null, contingency, mixture, drawn,
        classes, counts, counts,
        np.asarray([3, 4, 5]), np.asarray([4, 5, 3]), 0, 2,
    )
    path = tmp_path / "gradient_cross_partition.npz"
    cross_writer._write_cross_partition_artifact(path, arrays)

    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            assert data[key].dtype != object, key
        # Point estimates keep their own sentinels.
        assert float(data["c_same"]) == C_SAME
        assert float(data["c_different"]) == C_DIFF
        assert float(data["delta_cross"]) == DELTA_CROSS
        # The plug-in field is the plug-in value, not the map mean.
        assert float(data["mixture_median_similarity"]) == MIXTURE_PLUG_IN
        # Per-map vectors track their own statistic.
        assert float(data["c_same_per_map"][0]) == C_SAME
        assert float(data["c_different_per_map"][0]) == C_DIFF
        assert float(data["delta_cross_per_map"][0]) == DELTA_CROSS
        assert float(data["mixture_median_similarity_per_map"][0]) == MIXTURE_PLUG_IN
        # M = 1: unavailable, and NaN rather than None.
        assert int(data["map_count"]) == 1
        assert int(data["degrees_of_freedom"]) == 0
        assert bool(data["uncertainty_available"]) is False
        for name in ("c_same", "c_different", "delta_cross",
                     "mixture_median_similarity"):
            assert np.isnan(data[f"{name}_sd"]), name
            assert np.isnan(data[f"{name}_se"]), name


def test_sd_is_not_populated_from_se(tmp_path) -> None:
    """They differ by sqrt(M), so a swap cannot pass unnoticed."""

    import importlib as _il

    gc = _il.import_module("llm_behavior_lab.analysis.gradient_clustering")
    shared = _il.import_module("test_figure_uncertainty_artists")
    from llm_behavior_lab.analysis.gradient_cross_partition import (
        cross_partition_per_map,
    )

    record = shared._record(4)
    rows, usable = gc.unit_sketches_per_map(record)
    targets, greedy = shared._labels()
    result = cross_partition_per_map(rows[usable], targets[usable], greedy[usable])

    for name in ("c_same", "c_different", "delta_cross"):
        sd = result[name]["sample_sd"]
        se = result[name]["standard_error"]
        assert sd == pytest.approx(se * 2.0)
        assert sd != se


# -- M = 4: the production seam, end to end ------------------------------------------


def _m4_context():
    """A real b1 M=4 analysis result, plus the record and labels behind it."""

    import importlib as _il

    nc = _il.import_module("llm_behavior_lab.analysis.nucleus_clustering")
    shared = _il.import_module("test_figure_uncertainty_artists")

    record = shared._record(4)
    targets, greedy = shared._labels()
    # Asymmetric labels per temperature, so within/between/delta all differ.
    labels = np.stack([targets, greedy])
    temps = [0.6, 1.0]
    result = nc.nucleus_clustering(
        record, labels, temps, loss_temperatures=[0.6, 1.0],
        min_support=2, permutations=8,
    )
    return record, labels, result, temps


def test_the_nucleus_seam_round_trips_at_four_maps(tmp_path) -> None:
    """The exact builder and writer ``main`` calls, at M = 4."""

    import importlib as _il

    gc = _il.import_module("llm_behavior_lab.analysis.gradient_clustering")

    record, labels, result, temps = _m4_context()
    arrays = nucleus_writer._nucleus_artifact_arrays(result, record, labels)
    path = tmp_path / ARTIFACT_NAME
    nucleus_writer._write_nucleus_artifact(path, arrays)

    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            assert data[key].dtype != object, key

        assert int(data["map_count"]) == 4
        assert int(data["degrees_of_freedom"]) == 3
        assert bool(data["uncertainty_available"]) is True

        for name in ("delta_per_map", "within_per_map", "between_per_map",
                     "null_delta_per_map_mean"):
            assert data[name].shape == (len(temps), 4), name
        for grouping in ("target", "greedy"):
            assert data[f"reference_{grouping}_delta_per_map"].shape == (4,)

        # Every spread is a real number at M > 1.
        for name in ("delta", "within", "between",
                     "reference_target_delta", "reference_greedy_delta"):
            assert np.all(np.isfinite(data[f"{name}_sd"])), name
            assert np.all(np.isfinite(data[f"{name}_se"])), name

        # ddof = 1, and SE = SD / sqrt(4).
        for name in ("delta", "within", "between"):
            values = data[f"{name}_per_map"]
            assert np.allclose(data[f"{name}_sd"], values.std(axis=-1, ddof=1))
            assert np.allclose(data[f"{name}_se"], data[f"{name}_sd"] / 2.0)

        delta = data["delta_per_map"]
        within = data["within_per_map"]
        between = data["between_per_map"]
        assert np.allclose(delta, within - between, rtol=0, atol=1e-12)
        # Not pairwise copies of each other.
        assert not np.allclose(delta, within)
        assert not np.allclose(delta, between)
        assert not np.allclose(within, between)

    # The loader recovers the same values ...
    loaded = load_nucleus_clustering_artifact(tmp_path, ARTIFACT_NAME)
    assert loaded["map_count"] == 4
    assert loaded["degrees_of_freedom"] == 3
    assert loaded["uncertainty_available"] is True
    assert loaded["uncertainty"]["delta_per_map"].shape == (len(temps), 4)

    # ... and they equal independently invoked b1 APIs.
    for index, temperature in enumerate(temps):
        expected = gc.gradient_clustering_per_map(
            record, labels=labels[index], loss_temperature=float(temperature),
            min_support=2, permutations=8,
        )
        assert np.allclose(
            loaded["uncertainty"]["delta_per_map"][index],
            expected["delta_per_map"], rtol=0, atol=1e-12,
        )
        assert np.allclose(
            loaded["uncertainty"]["within_per_map"][index],
            expected["within_per_map"], rtol=0, atol=1e-12,
        )


def test_the_cross_partition_seam_round_trips_at_four_maps(tmp_path) -> None:
    """The exact builder and writer ``main`` calls, at M = 4."""

    import importlib as _il

    gc = _il.import_module("llm_behavior_lab.analysis.gradient_clustering")
    shared = _il.import_module("test_figure_uncertainty_artists")
    from llm_behavior_lab.analysis.gradient_cross_partition import (
        contingency_summary,
        cross_identity_null,
        cross_partition_matrix,
        cross_partition_per_map,
        mixture_reconstruction,
        pooled_cross_statistic,
    )

    record = shared._record(4)
    targets, greedy = shared._labels()
    rows, usable = gc.unit_sketches(record)
    rows, targets, greedy = rows[usable], targets[usable], greedy[usable]

    pooled = pooled_cross_statistic(rows, targets, greedy)
    null = cross_identity_null(rows, targets, greedy, permutations=8)
    contingency = contingency_summary(targets, greedy)
    mixture = mixture_reconstruction(rows, targets, greedy, min_support=2)
    shown = pooled["classes"]
    counts = np.bincount(targets, minlength=int(shown.max()) + 1)[shown]
    drawn = cross_partition_matrix(rows, targets, greedy, shown, shown)

    arrays = cross_writer._cross_partition_arrays(
        record, pooled, null, contingency, mixture, drawn,
        shown, counts, counts, targets, greedy, 0, 2,
    )
    path = tmp_path / "gradient_cross_partition.npz"
    cross_writer._write_cross_partition_artifact(path, arrays)

    per_map_rows, per_map_usable = gc.unit_sketches_per_map(record)
    expected = cross_partition_per_map(
        per_map_rows[per_map_usable], targets, greedy, min_support=2
    )

    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            assert data[key].dtype != object, key

        assert int(data["map_count"]) == 4
        assert int(data["degrees_of_freedom"]) == 3
        assert bool(data["uncertainty_available"]) is True

        for name in ("c_same", "c_different", "delta_cross",
                     "mixture_median_similarity"):
            assert data[f"{name}_per_map"].shape == (4,), name
            assert np.isfinite(data[f"{name}_sd"]), name
            assert np.isfinite(data[f"{name}_se"]), name
            values = data[f"{name}_per_map"]
            assert float(data[f"{name}_sd"]) == pytest.approx(values.std(ddof=1))
            # SD and SE differ by sqrt(4) = 2, so a swap cannot pass.
            assert float(data[f"{name}_sd"]) == pytest.approx(
                float(data[f"{name}_se"]) * 2.0
            )
            assert float(data[f"{name}_sd"]) != float(data[f"{name}_se"])

        assert np.allclose(
            data["delta_cross_per_map"],
            data["c_same_per_map"] - data["c_different_per_map"],
            rtol=0, atol=1e-15,
        )

        # The plug-in scalar is the ensemble value, not the mean of ratios.
        assert float(data["mixture_median_similarity"]) == pytest.approx(
            mixture["median_similarity"]
        )
        assert float(data["mixture_median_similarity_map_mean"]) == pytest.approx(
            float(np.mean(data["mixture_median_similarity_per_map"]))
        )
        assert float(data["mixture_median_similarity"]) != float(
            data["mixture_median_similarity_map_mean"]
        )

        # And the per-map values match independently invoked b1 output.
        assert np.allclose(
            data["c_same_per_map"], expected["c_same_per_map"], rtol=0, atol=1e-12
        )
        assert np.allclose(
            data["delta_cross_per_map"], expected["delta_cross_per_map"],
            rtol=0, atol=1e-12,
        )

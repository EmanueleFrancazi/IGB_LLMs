"""Tests for clustering gradients by the token nucleus sampling produced.

The estimator itself is already covered in ``tests/test_gradient_clustering.py``.
What has to be true *here* is that this sweep reuses that estimator rather than
re-deriving it, that the support diagnostics describe the same population the
statistic was computed over, and that the temperature axis cannot silently go
out of step with the label rows.

These are NumPy-only. Recovering the labels from a model needs PyTorch and is
covered in ``tests/test_nucleus_labels.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_behavior_lab.analysis import InitializationExperimentRecord
from llm_behavior_lab.analysis.gradient_clustering import gradient_clustering
from llm_behavior_lab.analysis.nucleus_clustering import (
    nucleus_clustering,
    support_diagnostics,
)

VOCAB = 14
ELIGIBLE = np.arange(2, VOCAB)
K = 32


#: Loss temperatures the multi-field fixture measures directions at.
LOSS_TEMPERATURES = (0.12, 0.24, 0.60, 1.0, 1.20)


def _record(
    num_positions: int = 90, seed: int = 7, *, loss_temperatures: bool = False
):
    """A record with sketches, optionally at several loss temperatures.

    When ``loss_temperatures`` is set, every T_g gets a *different* directional
    field, so a computation that read the wrong slice would give a different
    answer rather than quietly agreeing.
    """

    generator = np.random.default_rng(seed)
    rows = generator.normal(size=(num_positions, K))
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    targets = generator.choice(ELIGIBLE, size=num_positions)
    greedy = generator.choice(ELIGIBLE, size=num_positions)
    corpus = np.zeros(VOCAB, dtype=np.int64)
    corpus[ELIGIBLE] = 9
    return InitializationExperimentRecord.build(
        corpus_counts=corpus,
        selected_target_counts=np.bincount(targets, minlength=VOCAB),
        greedy_counts=np.bincount(greedy, minlength=VOCAB)[None, :],
        nucleus_counts=np.bincount(greedy, minlength=VOCAB)[None, None, :],
        mean_predicted_probabilities=np.full((1, VOCAB), 1.0 / VOCAB),
        model_seeds=[1000],
        eligible_token_ids=ELIGIBLE,
        metadata={
            "num_positions": num_positions,
            "analysis": {
                "num_positions": num_positions,
                "gradient_analysis": {
                    "enabled": True,
                    "initialization_index": 0,
                    "covers_all_positions": True,
                    "gradient_sketch": {"dimension": K, "seed": 1},
                },
            },
        },
        gradient_position_indices=np.arange(num_positions),
        gradient_position_target_ids=targets.astype(np.int64),
        gradient_position_greedy_ids=greedy.astype(np.int64),
        gradient_position_norms=np.ones(num_positions),
        gradient_position_sketches=rows,
        **_temperature_fields(rows, num_positions, generator, loss_temperatures),
    )


def _temperature_fields(rows, num_positions, generator, enabled):
    """The [N_T, D, K] field, with the canonical row equal to the canonical one."""

    if not enabled:
        return {}
    canonical = LOSS_TEMPERATURES.index(1.0)
    stacked = []
    for index in range(len(LOSS_TEMPERATURES)):
        if index == canonical:
            stacked.append(np.asarray(rows, dtype=np.float32))
            continue
        other = generator.normal(size=(num_positions, K))
        other /= np.linalg.norm(other, axis=1, keepdims=True)
        stacked.append(other.astype(np.float32))
    return {
        "gradient_temperatures": np.asarray(LOSS_TEMPERATURES),
        "gradient_temperature_position_norms": np.ones(
            (len(LOSS_TEMPERATURES), num_positions)
        ),
        "gradient_temperature_position_sketches": np.stack(stacked),
    }


# -- reuse of the estimator, not a parallel copy of it -----------------------


def test_supplying_the_greedy_labels_reproduces_the_greedy_grouping() -> None:
    """The label entry point must be the same estimator, not a lookalike.

    Handing it the very labels the record grouping would have derived has to
    give the identical pooled statistic, population and null -- otherwise the
    temperature trajectory and its two reference lines would be different
    statistics plotted on one axis.
    """

    record = _record()
    labels = np.asarray(record.gradient_position_greedy_ids)

    from_record = gradient_clustering(record, grouping="greedy", permutations=16)
    from_labels = gradient_clustering(
        record, grouping="greedy", labels=labels, permutations=16
    )

    for key in ("within", "between", "delta", "num_within_pairs", "num_between_pairs"):
        assert from_labels["population"][key] == from_record["population"][key]
    assert from_labels["null"]["delta_mean"] == from_record["null"]["delta_mean"]
    assert np.array_equal(
        from_labels["display"]["classes"], from_record["display"]["classes"]
    )


def test_the_sweep_calls_the_same_estimator_at_each_temperature() -> None:
    record = _record()
    greedy = np.asarray(record.gradient_position_greedy_ids)
    temperatures = [0.12, 0.60, 1.20]
    labels = np.stack([greedy, greedy, greedy])

    # T_g pinned, so the only thing that could differ between points is the
    # grouping -- and the labels are identical, so nothing may.
    result = nucleus_clustering(
        record, labels, temperatures, loss_temperatures=[1.0] * 3,
        permutations=16, display_classes=6,
    )

    # Identical labels, so only the per-temperature null seed may differ.
    deltas = {entry["population"]["delta"] for entry in result["by_temperature"]}
    assert len(deltas) == 1
    assert deltas.pop() == result["references"]["greedy"]["population"]["delta"]


def test_each_temperature_draws_its_own_permutation_sequence() -> None:
    """Sharing one seed would correlate the nulls along the trajectory."""

    record = _record()
    greedy = np.asarray(record.gradient_position_greedy_ids)
    result = nucleus_clustering(
        record, np.stack([greedy] * 3), [0.12, 0.60, 1.20],
        loss_temperatures=[1.0] * 3, permutations=16,
    )

    seeds = [entry["permutation_seed"] for entry in result["by_temperature"]]
    assert len(set(seeds)) == 3
    means = [entry["null"]["delta_mean"] for entry in result["by_temperature"]]
    assert len(set(means)) == 3


# -- the support diagnostics -------------------------------------------------


def test_support_diagnostics_match_explicit_counting() -> None:
    labels = np.array([5, 5, 5, 6, 6, 7, 8, 8, 8, 8], dtype=np.int64)

    summary = support_diagnostics(labels, min_support=2)

    assert summary["num_represented"] == 4
    assert summary["num_qualifying"] == 3          # 5, 6 and 8
    assert summary["num_singletons"] == 1          # 7
    assert summary["singleton_fraction"] == pytest.approx(0.25)
    assert summary["positions_in_qualifying"] == 9
    assert summary["fraction_positions_in_qualifying"] == pytest.approx(0.9)
    assert summary["largest_class"] == 4
    # 3*2 + 2*1 + 1*0 + 4*3
    assert summary["num_within_pairs"] == 20
    assert summary["num_between_pairs"] == 100 - (9 + 4 + 1 + 16)


def test_a_singleton_contributes_between_pairs_but_no_within_pair() -> None:
    """The all-position definition, which is what makes the sweep comparable.

    Dropping singletons would redefine "between" as "between non-singleton
    classes", and since the singleton fraction moves with temperature that
    redefinition would manufacture a trend from the support alone.
    """

    with_singleton = support_diagnostics(np.array([1, 1, 2, 2, 3]), min_support=2)
    without = support_diagnostics(np.array([1, 1, 2, 2]), min_support=2)

    assert with_singleton["num_within_pairs"] == without["num_within_pairs"]
    assert with_singleton["num_between_pairs"] > without["num_between_pairs"]


def test_the_diagnostics_describe_the_masked_population() -> None:
    """Zero-norm positions leave the statistic, so they must leave the counts."""

    record = _record(num_positions=40)
    norms = np.ones(40)
    norms[:5] = 0.0
    record = InitializationExperimentRecord.build(
        **{
            **{
                key: getattr(record, key)
                for key in (
                    "corpus_counts", "selected_target_counts", "greedy_counts",
                    "nucleus_counts", "mean_predicted_probabilities", "model_seeds",
                    "eligible_token_ids", "metadata", "gradient_position_indices",
                    "gradient_position_target_ids", "gradient_position_greedy_ids",
                    "gradient_position_sketches",
                )
            },
            "gradient_position_norms": norms,
        }
    )
    labels = np.asarray(record.gradient_position_greedy_ids)

    result = nucleus_clustering(
        record, labels[None, :], [0.6], loss_temperatures=[1.0], permutations=8
    )
    entry = result["by_temperature"][0]

    total = (
        entry["support"]["positions_in_qualifying"]
        + sum(
            count
            for count in np.unique(labels[5:], return_counts=True)[1]
            if count < 2
        )
    )
    assert total == 35
    assert entry["num_positions"] == 35


# -- the axes cannot go out of step ------------------------------------------


def test_a_label_row_per_temperature_is_required() -> None:
    record = _record()
    greedy = np.asarray(record.gradient_position_greedy_ids)

    with pytest.raises(ValueError, match="label rows"):
        nucleus_clustering(record, np.stack([greedy] * 2), [0.12, 0.60, 1.20])


def test_labels_must_cover_every_position() -> None:
    record = _record()
    greedy = np.asarray(record.gradient_position_greedy_ids)

    with pytest.raises(ValueError, match="positions"):
        nucleus_clustering(
            record, greedy[None, :-1], [0.6], loss_temperatures=[1.0]
        )


# -- the cached artifact must be indistinguishable from live results ---------


def _write_artifact(result, directory):
    """The writer's array layout, without its model-dependent front half."""

    import numpy as np

    arrays = {
        "temperatures": np.asarray(result["temperatures"], dtype=float),
        "num_positions": np.array(result["by_temperature"][0]["num_positions"]),
        "num_positions_excluded": np.array(
            result["by_temperature"][0]["num_positions_excluded"]
        ),
        "sketch_dimension": np.array(result["by_temperature"][0]["sketch_dimension"]),
        "min_support": np.array(result["min_support"]),
        "permutations": np.array(result["permutations"]),
        "permutation_seed": np.array(result["permutation_seed"]),
        "nucleus_labels": np.zeros(
            (len(result["temperatures"]), result["by_temperature"][0]["num_positions"]),
            dtype=np.int64,
        ),
        "delta": np.array([e["population"]["delta"] for e in result["by_temperature"]]),
        "within": np.array([e["population"]["within"] for e in result["by_temperature"]]),
        "between": np.array(
            [e["population"]["between"] for e in result["by_temperature"]]
        ),
        "null_delta_mean": np.array(
            [e["null"]["delta_mean"] for e in result["by_temperature"]]
        ),
        "null_delta_low": np.array(
            [e["null"]["delta_low"] for e in result["by_temperature"]]
        ),
        "null_delta_high": np.array(
            [e["null"]["delta_high"] for e in result["by_temperature"]]
        ),
    }
    for field in (
        "num_represented", "num_qualifying", "num_singletons", "singleton_fraction",
        "positions_in_qualifying", "fraction_positions_in_qualifying",
        "largest_class", "median_class", "num_within_pairs", "num_between_pairs",
    ):
        arrays[f"support_{field}"] = np.array(
            [entry["support"][field] for entry in result["by_temperature"]]
        )
    for index, entry in enumerate(result["by_temperature"]):
        arrays[f"display_classes_{index}"] = entry["display"]["classes"]
        arrays[f"display_matrix_{index}"] = entry["display"]["matrix"]
        arrays[f"display_counts_{index}"] = entry["display"]["counts"]
    for grouping in ("target", "greedy"):
        reference = result["references"][grouping]
        arrays[f"reference_{grouping}_delta"] = np.array(
            reference["population"]["delta"]
        )
        arrays[f"reference_{grouping}_null_low"] = np.array(
            reference["null"]["delta_low"]
        )
        arrays[f"reference_{grouping}_null_high"] = np.array(
            reference["null"]["delta_high"]
        )
    np.savez_compressed(directory / "nucleus_gradient_clustering.npz", **arrays)


def test_the_cached_artifact_round_trips_everything_the_figure_reads(tmp_path) -> None:
    """Anything the loader invents instead of storing is a way for a figure to
    show numbers the analysis never produced. Checked field by field."""

    from llm_behavior_lab.analysis.nucleus_clustering_artifact import (
        load_nucleus_clustering_artifact,
    )

    record = _record()
    greedy = np.asarray(record.gradient_position_greedy_ids)
    generator = np.random.default_rng(3)
    labels = np.stack([
        np.where(generator.random(greedy.size) < fraction,
                 generator.choice(ELIGIBLE, size=greedy.size), greedy)
        for fraction in (0.0, 0.4, 0.9)
    ])
    live = nucleus_clustering(
        record, labels, [0.12, 0.60, 1.20], loss_temperatures=[1.0] * 3,
        permutations=16, display_classes=5,
    )
    _write_artifact(live, tmp_path)

    loaded = load_nucleus_clustering_artifact(tmp_path)

    assert loaded["temperatures"] == live["temperatures"]
    assert loaded["min_support"] == live["min_support"]
    assert loaded["permutations"] == live["permutations"]
    assert loaded["permutation_seed"] == live["permutation_seed"]
    for stored, computed in zip(loaded["by_temperature"], live["by_temperature"]):
        assert stored["temperature"] == computed["temperature"]
        assert stored["num_positions"] == computed["num_positions"]
        assert stored["sketch_dimension"] == computed["sketch_dimension"]
        for key in ("delta", "within", "between"):
            assert stored["population"][key] == pytest.approx(
                computed["population"][key], abs=1e-12
            )
        for key in ("delta_mean", "delta_low", "delta_high"):
            assert stored["null"][key] == pytest.approx(
                computed["null"][key], abs=1e-12
            )
        assert stored["support"] == computed["support"]
        assert np.array_equal(
            stored["display"]["classes"], computed["display"]["classes"]
        )
        assert np.allclose(
            stored["display"]["matrix"], computed["display"]["matrix"],
            equal_nan=True,
        )
    for grouping in ("target", "greedy"):
        assert loaded["references"][grouping]["population"]["delta"] == pytest.approx(
            live["references"][grouping]["population"]["delta"], abs=1e-12
        )


def test_the_sketch_width_is_stored_not_read_off_the_heatmap(tmp_path) -> None:
    """The displayed matrix is class-by-class, so its shape says nothing about K.

    Built so the two would disagree: five displayed classes against a sketch of
    width 32. Inferring one from the other would report K = 5.
    """

    from llm_behavior_lab.analysis.nucleus_clustering_artifact import (
        load_nucleus_clustering_artifact,
    )

    record = _record()
    greedy = np.asarray(record.gradient_position_greedy_ids)
    live = nucleus_clustering(
        record, greedy[None, :], [0.6], loss_temperatures=[1.0],
        permutations=8, display_classes=5,
    )
    _write_artifact(live, tmp_path)

    loaded = load_nucleus_clustering_artifact(tmp_path)

    assert loaded["by_temperature"][0]["display"]["matrix"].shape == (5, 5)
    assert loaded["by_temperature"][0]["sketch_dimension"] == K


def test_a_missing_artifact_names_the_writer(tmp_path) -> None:
    from llm_behavior_lab.analysis.nucleus_clustering_artifact import (
        load_nucleus_clustering_artifact,
    )

    with pytest.raises(FileNotFoundError, match="write_nucleus_clustering_artifact"):
        load_nucleus_clustering_artifact(tmp_path)


# -- paired sampling and loss temperatures -----------------------------------


def test_a_sweep_defaults_the_loss_temperature_to_the_sampling_one() -> None:
    record = _record(loss_temperatures=True)
    greedy = np.asarray(record.gradient_position_greedy_ids)

    result = nucleus_clustering(
        record, np.stack([greedy] * 2), [0.12, 0.60], permutations=8
    )

    assert np.array_equal(result["loss_temperatures"], [0.12, 0.60])
    assert result["loss_defaulted"] is True
    for entry, expected in zip(result["by_temperature"], (0.12, 0.60)):
        assert entry["sampling_temperature"] == expected
        assert entry["loss_temperature"] == expected


def test_the_controlled_design_pins_the_loss_temperature(tmp_path) -> None:
    """Figure 22's sweep: T_s varies, T_g stays at the record's gradients."""

    record = _record()
    greedy = np.asarray(record.gradient_position_greedy_ids)
    sampling = [0.12, 0.24, 0.60]

    result = nucleus_clustering(
        record, np.stack([greedy] * 3), sampling,
        loss_temperatures=[1.0, 1.0, 1.0], permutations=8,
    )

    assert np.array_equal(result["sampling_temperatures"], sampling)
    assert np.all(result["loss_temperatures"] == 1.0)
    assert result["loss_defaulted"] is False
    for entry in result["by_temperature"]:
        assert entry["loss_temperature"] == 1.0
        assert "T_s=" in entry["grouping"] and "T_g=" in entry["grouping"]


def test_the_sampling_temperatures_stay_available_under_the_old_key() -> None:
    """Readers written before the split looked up "temperatures"."""

    record = _record()
    greedy = np.asarray(record.gradient_position_greedy_ids)

    result = nucleus_clustering(
        record, np.stack([greedy] * 2), [0.12, 0.60],
        loss_temperatures=[1.0, 1.0], permutations=8,
    )

    assert result["temperatures"] == (0.12, 0.60)


def test_a_repeated_pair_is_measured_once_and_reused() -> None:
    """A sweep pinning T_g must not pay for the same measurement twice.

    Both coordinates repeat here, so the second pair is the same measurement as
    the first -- same labels, same gradient directions -- and reusing it is
    exact rather than approximate.
    """

    record = _record()
    greedy = np.asarray(record.gradient_position_greedy_ids)
    labels = np.stack([greedy, greedy, greedy])

    result = nucleus_clustering(
        record, labels, [0.12, 0.12, 0.60],
        loss_temperatures=[1.0, 1.0, 1.0], permutations=8,
    )

    assert result["num_pairs_reused"] == 1
    first, repeat, other = result["by_temperature"]
    assert repeat["reused_from_pair"] == 0
    assert first["reused_from_pair"] is None
    assert other["reused_from_pair"] is None
    assert repeat["population"]["delta"] == first["population"]["delta"]
    assert repeat["null"]["delta_mean"] == first["null"]["delta_mean"]
    # The reused entry still reports its own place in the sweep.
    assert repeat["pair_index"] == 1


def test_pairs_differing_in_either_coordinate_are_both_computed() -> None:
    """Reuse must key on the pair, not on one half of it."""

    record = _record(loss_temperatures=True)
    greedy = np.asarray(record.gradient_position_greedy_ids)

    same_sampling = nucleus_clustering(
        record, np.stack([greedy] * 2), [0.12, 0.12],
        loss_temperatures=[1.0, 0.12], permutations=8,
    )
    assert same_sampling["num_pairs_reused"] == 0

    same_loss = nucleus_clustering(
        record, np.stack([greedy] * 2), [0.12, 0.60],
        loss_temperatures=[1.0, 1.0], permutations=8,
    )
    assert same_loss["num_pairs_reused"] == 0


def test_a_mismatched_pair_length_is_rejected() -> None:
    record = _record()
    greedy = np.asarray(record.gradient_position_greedy_ids)

    with pytest.raises(ValueError, match="same length"):
        nucleus_clustering(
            record, np.stack([greedy] * 2), [0.12, 0.60], loss_temperatures=[1.0]
        )


def test_a_non_positive_loss_temperature_is_rejected() -> None:
    record = _record()
    greedy = np.asarray(record.gradient_position_greedy_ids)

    with pytest.raises(ValueError, match="loss_temperatures must be positive"):
        nucleus_clustering(
            record, np.stack([greedy] * 2), [0.12, 0.60], loss_temperatures=[1.0, 0.0]
        )


def test_an_unmeasured_loss_temperature_fails_rather_than_using_the_canonical() -> None:
    """The whole point of the field: no silent substitution."""

    record = _record(loss_temperatures=True)
    greedy = np.asarray(record.gradient_position_greedy_ids)

    with pytest.raises(ValueError, match="No measured gradient-direction field"):
        nucleus_clustering(
            record, np.stack([greedy] * 2), [0.12, 0.60],
            loss_temperatures=[1.0, 0.36], permutations=8,
        )


# -- reading artifacts from both eras ----------------------------------------


def _write_paired_artifact(result, directory):
    """The writer's layout including the explicit pair arrays."""

    import numpy as np

    _write_artifact(result, directory)
    path = directory / "nucleus_gradient_clustering.npz"
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["sampling_temperatures"] = np.asarray(
        result["sampling_temperatures"], dtype=float
    )
    arrays["loss_temperatures"] = np.asarray(result["loss_temperatures"], dtype=float)
    np.savez_compressed(path, **arrays)


def test_an_artifact_from_before_the_split_keeps_its_original_meaning(tmp_path) -> None:
    """Figure 22's real artifact stores only "temperatures".

    It must keep reading as the controlled design it was -- T_s varying against
    the canonical T_g = 1 gradients -- rather than being reinterpreted as the
    new matched default.
    """

    from llm_behavior_lab.analysis.nucleus_clustering_artifact import (
        load_nucleus_clustering_artifact,
    )

    record = _record()
    greedy = np.asarray(record.gradient_position_greedy_ids)
    live = nucleus_clustering(
        record, np.stack([greedy] * 3), [0.12, 0.60, 1.20],
        loss_temperatures=[1.0, 1.0, 1.0], permutations=8, display_classes=5,
    )
    _write_artifact(live, tmp_path)          # old layout: no pair arrays

    loaded = load_nucleus_clustering_artifact(tmp_path)

    assert loaded["pair_metadata"] == "historical_fallback"
    assert np.array_equal(loaded["sampling_temperatures"], [0.12, 0.60, 1.20])
    assert np.all(loaded["loss_temperatures"] == 1.0)
    assert loaded["pairing"] == "T_g = 1 fixed"
    # And the values it reports are the ones that were stored.
    for stored, computed in zip(loaded["by_temperature"], live["by_temperature"]):
        assert stored["population"]["delta"] == pytest.approx(
            computed["population"]["delta"], abs=1e-12
        )


def test_an_explicit_artifact_round_trips_both_temperature_arrays(tmp_path) -> None:
    from llm_behavior_lab.analysis.nucleus_clustering_artifact import (
        load_nucleus_clustering_artifact,
    )

    record = _record(loss_temperatures=True)
    greedy = np.asarray(record.gradient_position_greedy_ids)
    live = nucleus_clustering(
        record, np.stack([greedy] * 3), [0.12, 0.60, 1.20],
        loss_temperatures=[1.0, 0.12, 0.24], permutations=8, display_classes=5,
    )
    _write_paired_artifact(live, tmp_path)

    loaded = load_nucleus_clustering_artifact(tmp_path)

    assert loaded["pair_metadata"] == "explicit"
    assert np.array_equal(loaded["sampling_temperatures"], [0.12, 0.60, 1.20])
    assert np.array_equal(loaded["loss_temperatures"], [1.0, 0.12, 0.24])
    assert loaded["pairing"] == "T_g varies independently of T_s"
    for index, entry in enumerate(loaded["by_temperature"]):
        assert entry["loss_temperature"] == [1.0, 0.12, 0.24][index]


def test_a_matched_artifact_is_described_as_matched(tmp_path) -> None:
    from llm_behavior_lab.analysis.nucleus_clustering_artifact import (
        load_nucleus_clustering_artifact,
    )

    record = _record(loss_temperatures=True)
    greedy = np.asarray(record.gradient_position_greedy_ids)
    live = nucleus_clustering(
        record, np.stack([greedy] * 3), [0.12, 0.60, 1.20],
        permutations=8, display_classes=5,
    )
    _write_paired_artifact(live, tmp_path)

    loaded = load_nucleus_clustering_artifact(tmp_path)

    assert loaded["pairing"] == "T_g = T_s (matched)"


def test_a_result_array_out_of_step_with_the_pairs_is_caught(tmp_path) -> None:
    """The reader validates rather than plotting one pair's value on another."""

    from llm_behavior_lab.analysis.nucleus_clustering_artifact import (
        load_nucleus_clustering_artifact,
    )

    record = _record()
    greedy = np.asarray(record.gradient_position_greedy_ids)
    live = nucleus_clustering(
        record, np.stack([greedy] * 3), [0.12, 0.60, 1.20],
        loss_temperatures=[1.0, 1.0, 1.0], permutations=8, display_classes=5,
    )
    _write_paired_artifact(live, tmp_path)

    path = tmp_path / "nucleus_gradient_clustering.npz"
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["delta"] = arrays["delta"][:2]
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError, match="leading dimension"):
        load_nucleus_clustering_artifact(tmp_path)


# -- the pair must reach the right gradient field ----------------------------


def _planted_record(num_positions: int = 60, seed: int = 31):
    """Structure at one loss temperature and none at another.

    Built so the answer differs by construction: at T_g = 0.12 the gradients are
    strongly clustered by greedy token, at T_g = 1 they are pure noise. A
    computation that silently read the canonical field for a T_g = 0.12 request
    cannot produce the clustered answer, so the substitution this guards against
    is visible in Delta rather than hidden in the last digits.
    """

    generator = np.random.default_rng(seed)
    greedy = generator.choice(ELIGIBLE[:4], size=num_positions)
    targets = generator.choice(ELIGIBLE, size=num_positions)

    def noise():
        rows = generator.normal(size=(num_positions, K))
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)

    centres = generator.normal(size=(VOCAB, K))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    clustered = centres[greedy] + 0.05 * noise()
    clustered /= np.linalg.norm(clustered, axis=1, keepdims=True)

    canonical = LOSS_TEMPERATURES.index(1.0)
    stacked = []
    for index, value in enumerate(LOSS_TEMPERATURES):
        if index == canonical:
            stacked.append(noise().astype(np.float32))       # no structure
        elif value == 0.12:
            stacked.append(clustered.astype(np.float32))     # strong structure
        else:
            stacked.append(noise().astype(np.float32))
    stacked = np.stack(stacked)

    corpus = np.zeros(VOCAB, dtype=np.int64)
    corpus[ELIGIBLE] = 9
    return InitializationExperimentRecord.build(
        corpus_counts=corpus,
        selected_target_counts=np.bincount(targets, minlength=VOCAB),
        greedy_counts=np.bincount(greedy, minlength=VOCAB)[None, :],
        nucleus_counts=np.bincount(greedy, minlength=VOCAB)[None, None, :],
        mean_predicted_probabilities=np.full((1, VOCAB), 1.0 / VOCAB),
        model_seeds=[1000],
        eligible_token_ids=ELIGIBLE,
        metadata={
            "num_positions": num_positions,
            "analysis": {
                "num_positions": num_positions,
                "gradient_analysis": {
                    "enabled": True,
                    "initialization_index": 0,
                    "covers_all_positions": True,
                    "gradient_sketch": {"dimension": K, "seed": 1},
                },
            },
        },
        gradient_position_indices=np.arange(num_positions),
        gradient_position_target_ids=targets.astype(np.int64),
        gradient_position_greedy_ids=greedy.astype(np.int64),
        gradient_position_norms=np.ones(num_positions),
        gradient_position_sketches=stacked[canonical],
        gradient_temperatures=np.asarray(LOSS_TEMPERATURES),
        gradient_temperature_position_norms=np.ones(
            (len(LOSS_TEMPERATURES), num_positions)
        ),
        gradient_temperature_position_sketches=stacked,
    )


def test_the_same_labels_at_two_loss_temperatures_give_different_geometry() -> None:
    record = _planted_record()
    greedy = np.asarray(record.gradient_position_greedy_ids)

    result = nucleus_clustering(
        record, np.stack([greedy, greedy]), [0.60, 0.60],
        loss_temperatures=[0.12, 1.0], permutations=16,
    )

    structured, unstructured = result["by_temperature"]
    assert structured["loss_temperature"] == 0.12
    assert unstructured["loss_temperature"] == 1.0
    # Planted structure at 0.12, noise at 1. Reading the canonical field for the
    # first pair would collapse this gap.
    assert structured["population"]["delta"] > 0.5
    assert abs(unstructured["population"]["delta"]) < 0.1
    assert result["num_fields_read"] == 2


def test_different_labels_at_one_loss_temperature_share_the_gradient_field() -> None:
    record = _planted_record()
    greedy = np.asarray(record.gradient_position_greedy_ids)
    targets = np.asarray(record.gradient_position_target_ids)

    result = nucleus_clustering(
        record, np.stack([greedy, targets]), [0.12, 0.60],
        loss_temperatures=[0.12, 0.12], permutations=16,
    )

    assert result["num_fields_read"] == 1
    assert result["num_pairs_reused"] == 0          # different labels
    by_greedy, by_target = result["by_temperature"]
    # Same field, different groupings: the greedy grouping is the planted one.
    assert by_greedy["population"]["delta"] > by_target["population"]["delta"]


def test_a_pinned_loss_temperature_reads_one_field_for_every_point() -> None:
    record = _planted_record()
    greedy = np.asarray(record.gradient_position_greedy_ids)

    result = nucleus_clustering(
        record, np.stack([greedy] * 3), [0.12, 0.24, 0.60],
        loss_temperatures=[1.0] * 3, permutations=8,
    )

    assert result["num_fields_read"] == 1
    assert result["num_unique_loss"] == 1


def test_the_references_come_from_the_pairs_own_loss_temperature() -> None:
    """A reference from a different gradient field is a different geometry."""

    record = _planted_record()
    greedy = np.asarray(record.gradient_position_greedy_ids)

    result = nucleus_clustering(
        record, np.stack([greedy, greedy]), [0.60, 0.60],
        loss_temperatures=[0.12, 1.0], permutations=16,
    )

    by_loss = result["references_by_loss_temperature"]
    assert set(by_loss) == {0.12, 1.0}
    # The greedy grouping is the planted structure, so its reference must be
    # large in the structured field and small in the noise one.
    assert by_loss[0.12]["greedy"]["population"]["delta"] > 0.5
    assert abs(by_loss[1.0]["greedy"]["population"]["delta"]) < 0.1
    for grouping in ("target", "greedy"):
        for value, reference in by_loss.items():
            assert reference[grouping]["loss_temperature"] == value


def test_the_per_pair_null_preserves_that_pairs_label_histogram() -> None:
    """Permuting labels must not change the class structure it permutes."""

    record = _planted_record()
    greedy = np.asarray(record.gradient_position_greedy_ids)

    result = nucleus_clustering(
        record, np.stack([greedy]), [0.60], loss_temperatures=[0.12],
        permutations=32,
    )
    entry = result["by_temperature"][0]

    # The null is computed over the same population the statistic is, so the
    # counts it permutes are exactly this T_s histogram.
    observed = np.unique(greedy, return_counts=True)[1]
    assert np.array_equal(
        np.sort(entry["population"]["counts"]), np.sort(observed)
    )
    assert entry["null"]["permutations"] == 32
    # Planted structure sits far outside the shuffled reference.
    assert entry["population"]["delta"] > entry["null"]["delta_high"]


def test_the_pair_artifact_carries_references_per_loss_temperature(tmp_path) -> None:
    """Scalar references are ambiguous once T_g varies, so the axis is stored."""

    from llm_behavior_lab.analysis.nucleus_clustering_artifact import (
        load_nucleus_clustering_artifact,
    )

    record = _planted_record()
    greedy = np.asarray(record.gradient_position_greedy_ids)
    live = nucleus_clustering(
        record, np.stack([greedy, greedy]), [0.60, 0.60],
        loss_temperatures=[0.12, 1.0], permutations=8, display_classes=4,
    )
    _write_paired_artifact(live, tmp_path)

    # The writer's reference-axis arrays, which _write_paired_artifact omits.
    path = tmp_path / "nucleus_gradient_clustering.npz"
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    losses = sorted(live["references_by_loss_temperature"])
    arrays["reference_loss_temperatures"] = np.asarray(losses, dtype=float)
    for grouping in ("target", "greedy"):
        for field, keys in (
            ("delta", ("population", "delta")),
            ("null_low", ("null", "delta_low")),
            ("null_high", ("null", "delta_high")),
        ):
            arrays[f"reference_by_loss_{grouping}_{field}"] = np.asarray(
                [
                    live["references_by_loss_temperature"][value][grouping][keys[0]][keys[1]]
                    for value in losses
                ],
                dtype=float,
            )
    np.savez_compressed(path, **arrays)

    loaded = load_nucleus_clustering_artifact(tmp_path)

    by_loss = loaded["references_by_loss_temperature"]
    assert sorted(by_loss) == [0.12, 1.0]
    for value in (0.12, 1.0):
        for grouping in ("target", "greedy"):
            assert by_loss[value][grouping]["population"]["delta"] == pytest.approx(
                live["references_by_loss_temperature"][value][grouping]["population"][
                    "delta"
                ],
                abs=1e-12,
            )


def test_an_artifact_without_the_reference_axis_still_loads(tmp_path) -> None:
    """Figure 22's real artifact predates it and must keep working."""

    from llm_behavior_lab.analysis.nucleus_clustering_artifact import (
        load_nucleus_clustering_artifact,
    )

    record = _record()
    greedy = np.asarray(record.gradient_position_greedy_ids)
    live = nucleus_clustering(
        record, np.stack([greedy] * 2), [0.12, 0.60], loss_temperatures=[1.0, 1.0],
        permutations=8, display_classes=5,
    )
    _write_artifact(live, tmp_path)

    loaded = load_nucleus_clustering_artifact(tmp_path)

    assert loaded["references_by_loss_temperature"] == {}
    assert loaded["pair_metadata"] == "historical_fallback"
    assert loaded["references"]["greedy"]["population"]["delta"] == pytest.approx(
        live["references"]["greedy"]["population"]["delta"], abs=1e-12
    )


# -- which artifact each design writes to -------------------------------------
#
# The control and the matched design must coexist rather than overwrite each
# other. Everything else the CLI already accepts has to keep working exactly as
# before: this phase adds a name, it does not narrow what may be requested.


def _artifact_name_for():
    """The writer's naming rule, loaded without importing PyTorch."""

    import ast
    import pathlib

    script = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts"
        / "write_nucleus_clustering_artifact.py"
    )
    tree = ast.parse(script.read_text())
    wanted = {"artifact_name_for"}
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert functions, "artifact_name_for is missing from the writer"
    namespace = {
        "np": np,
        "ARTIFACT_NAME": "nucleus_gradient_clustering.npz",
        "MATCHED_ARTIFACT_NAME": "nucleus_gradient_clustering_matched_TsTg.npz",
    }
    exec(  # noqa: S102 - reading one pure function out of a torch-importing script
        compile(ast.Module(body=functions, type_ignores=[]), "<writer>", "exec"),
        namespace,
    )
    return namespace["artifact_name_for"]


SWEEP = [0.12, 0.24, 0.36, 0.48, 0.60, 1.20]


def test_the_control_keeps_the_canonical_artifact_name() -> None:
    """Figure 22 reads this name and must keep finding the control there."""

    assert _artifact_name_for()(SWEEP, [1.0] * 6) == "nucleus_gradient_clustering.npz"


def test_the_matched_design_gets_its_own_artifact_name() -> None:
    assert _artifact_name_for()(SWEEP, SWEEP) == (
        "nucleus_gradient_clustering_matched_TsTg.npz"
    )


def test_an_arbitrary_explicit_pairing_is_still_accepted() -> None:
    """No CLI regression.

    --loss-temperatures already accepts any explicit equal-length list. Such a
    pairing is neither the control nor the matched design; it must keep working
    and keep the name it has always written to, rather than being refused for
    being awkward to name.
    """

    name = _artifact_name_for()([0.12, 0.60], [1.0, 0.12])

    assert name == "nucleus_gradient_clustering.npz"


def test_a_single_point_matched_sweep_is_still_matched() -> None:
    assert _artifact_name_for()([0.36], [0.36]) == (
        "nucleus_gradient_clustering_matched_TsTg.npz"
    )


def test_the_loader_reads_whichever_artifact_it_is_given(tmp_path) -> None:
    from llm_behavior_lab.analysis.nucleus_clustering_artifact import (
        ARTIFACT_NAME,
        MATCHED_ARTIFACT_NAME,
        load_nucleus_clustering_artifact,
    )

    record = _record(loss_temperatures=True)
    greedy = np.asarray(record.gradient_position_greedy_ids)
    control = nucleus_clustering(
        record, np.stack([greedy] * 2), [0.12, 0.60],
        loss_temperatures=[1.0, 1.0], permutations=8, display_classes=5,
    )
    matched = nucleus_clustering(
        record, np.stack([greedy] * 2), [0.12, 0.60],
        loss_temperatures=[0.12, 0.60], permutations=8, display_classes=5,
    )
    # The paired layout, so both temperature arrays are actually stored.
    _write_paired_artifact(control, tmp_path)
    (tmp_path / ARTIFACT_NAME).rename(tmp_path / "control_tmp.npz")
    _write_paired_artifact(matched, tmp_path)
    (tmp_path / ARTIFACT_NAME).rename(tmp_path / MATCHED_ARTIFACT_NAME)
    (tmp_path / "control_tmp.npz").rename(tmp_path / ARTIFACT_NAME)

    default = load_nucleus_clustering_artifact(tmp_path)
    explicit = load_nucleus_clustering_artifact(tmp_path, MATCHED_ARTIFACT_NAME)

    assert np.all(default["loss_temperatures"] == 1.0)
    assert np.array_equal(
        explicit["loss_temperatures"], explicit["sampling_temperatures"]
    )

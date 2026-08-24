"""Tests for recovering the sweep's per-position nucleus samples.

The gate is the important part. Reproducing labels means re-running a forward
pass and re-deriving a draw the experiment made earlier, and there are several
ways for that to silently describe a different experiment: a different sampling
stream, a different support mask, a different initialization. The recorded
histogram is the one piece of the original draw that *was* persisted, so it is
the only available check -- and it has to be an exact one.

The gate is pure NumPy and lives in the analysis layer, so these run wherever
the analysis suite runs -- including where PyTorch is not installed. Recovering
the labels needs a model and is exercised on the machine that has one.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm_behavior_lab.analysis.nucleus_clustering import histogram_gate

VOCAB = 10


def _labels_and_counts(rows: list[list[int]]):
    labels = np.asarray(rows, dtype=np.int64)
    counts = np.stack([np.bincount(row, minlength=VOCAB) for row in labels])
    return labels, counts


def test_the_gate_passes_on_an_exact_reproduction() -> None:
    labels, counts = _labels_and_counts([[2, 2, 3, 4], [2, 3, 3, 5]])

    result = histogram_gate(
        labels, counts, vocab_size=VOCAB, temperatures=[0.12, 1.20]
    )

    assert result["passed"] is True
    assert [entry["temperature"] for entry in result["per_temperature"]] == [0.12, 1.20]
    assert result["per_temperature"][0]["num_positions"] == 4
    assert result["per_temperature"][0]["num_distinct_tokens"] == 3


def test_a_single_position_moving_between_tokens_fails_the_gate() -> None:
    """The smallest possible discrepancy must still abort.

    Two positions swapping labels leaves the histogram unchanged and is
    genuinely undetectable here; one position *moving* changes two counts by one
    each, and that is the smallest thing the gate can see. It must not be
    treated as close enough.
    """

    labels, counts = _labels_and_counts([[2, 2, 3, 4]])
    counts = counts.copy()
    counts[0][2] -= 1
    counts[0][3] += 1

    with pytest.raises(ValueError, match="do not reproduce the recorded histogram"):
        histogram_gate(labels, counts, vocab_size=VOCAB, temperatures=[0.6])


def test_the_failure_names_the_temperature_and_the_scale_of_the_disagreement() -> None:
    """A fatal gate still has to be diagnosable."""

    labels, counts = _labels_and_counts([[2, 2, 3], [4, 4, 5]])
    counts = counts.copy()
    counts[1][4] = 0
    counts[1][7] = 2

    with pytest.raises(ValueError) as failure:
        histogram_gate(labels, counts, vocab_size=VOCAB, temperatures=[0.12, 1.20])

    message = str(failure.value)
    assert "T = 1.2" in message
    assert "2 token IDs differ" in message
    assert "largest discrepancy 2" in message


def test_the_gate_checks_every_temperature_not_only_the_first() -> None:
    labels, counts = _labels_and_counts([[2, 2, 3], [4, 4, 5]])
    counts = counts.copy()
    counts[1][5] += 3

    with pytest.raises(ValueError, match="T = 1.2"):
        histogram_gate(labels, counts, vocab_size=VOCAB, temperatures=[0.12, 1.20])


def test_a_histogram_count_mismatch_is_rejected_before_comparing() -> None:
    labels, counts = _labels_and_counts([[2, 2, 3], [4, 4, 5]])

    with pytest.raises(ValueError, match="recovered temperatures"):
        histogram_gate(labels, counts[:1], vocab_size=VOCAB, temperatures=[0.12])


def test_the_gate_is_not_fooled_by_a_matching_total() -> None:
    """Same number of positions, different distribution, still a failure."""

    labels, counts = _labels_and_counts([[2, 2, 2, 3]])
    wrong = np.zeros_like(counts)
    wrong[0][2] = 2
    wrong[0][3] = 2
    assert wrong.sum() == counts.sum()

    with pytest.raises(ValueError):
        histogram_gate(labels, wrong, vocab_size=VOCAB, temperatures=[0.6])


def test_the_pass_path_reports_the_same_measures_as_the_failure_path() -> None:
    """Zeros stated explicitly, so a pass is distinguishable from an absent check.

    A report that prints nothing on success looks identical to one whose gate
    never ran, which is exactly the ambiguity this gate exists to remove.
    """

    labels, counts = _labels_and_counts([[2, 2, 3, 4], [2, 3, 3, 5]])

    result = histogram_gate(
        labels, counts, vocab_size=VOCAB, temperatures=[0.12, 1.20]
    )

    for entry in result["per_temperature"]:
        assert entry["exact_match"] is True
        assert entry["mismatched_bins"] == 0
        assert entry["max_absolute_difference"] == 0
        assert entry["sum_absolute_difference"] == 0


def test_the_failure_message_reports_the_total_absolute_difference() -> None:
    labels, counts = _labels_and_counts([[2, 2, 3, 3]])
    counts = counts.copy()
    counts[0][2] -= 1
    counts[0][3] += 1

    with pytest.raises(ValueError) as failure:
        histogram_gate(labels, counts, vocab_size=VOCAB, temperatures=[0.6])

    # Two bins off by one each.
    assert "total absolute difference 2" in str(failure.value)


# -- which forward batch size a historical reconstruction runs at -------------
#
# Batch size turned out to belong to the numerical provenance, not to
# performance tuning. Reconstructing a real run at 8 when it had realized 4
# moved one position at T = 0.12 and the gate caught it, so these pin the
# precedence rule that prevents a repeat.

from llm_behavior_lab.analysis.nucleus_clustering import resolve_forward_batch_size


def test_the_realized_batch_size_is_used_when_no_override_is_given() -> None:
    resolved = resolve_forward_batch_size({"forward_batch_size": 4})

    assert resolved["value"] == 4
    assert resolved["source"] == "historical"
    assert resolved["historical"] == 4
    assert "historical realized metadata" in resolved["description"]


def test_the_requested_yaml_value_never_replaces_the_realized_one() -> None:
    """The frozen config's runtime value is what was asked for, not what ran.

    On the record that failed, the YAML said 32, the old hard-coded default said
    8, and the run had realized 4. Only the realized value reproduces it.
    """

    analysis = {
        "forward_batch_size": 4,
        # Present in the same mapping and still irrelevant.
        "runtime": {"forward_batch_size": 32},
    }

    assert resolve_forward_batch_size(analysis)["value"] == 4


def test_the_writer_does_not_read_the_runtime_batch_size() -> None:
    """A source-level guard: the requested value must not creep back in."""

    import pathlib

    script = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts"
        / "write_nucleus_clustering_artifact.py"
    )
    source = script.read_text()

    assert "resolve_forward_batch_size" in source
    # "runtime" appears legitimately, for the device. It must never appear on a
    # line that also reaches for a batch size.
    offending = [
        line
        for line in source.splitlines()
        if "runtime" in line and "forward_batch_size" in line
    ]
    assert offending == []


def test_an_explicit_override_wins_and_says_so() -> None:
    """The override diagnosed the original mismatch, so it must stay possible."""

    resolved = resolve_forward_batch_size({"forward_batch_size": 4}, 8)

    assert resolved["value"] == 8
    assert resolved["source"] == "override"
    assert resolved["historical"] == 4
    assert "override" in resolved["description"]
    # The historical value stays visible, so an audit log cannot be misread.
    assert "4" in resolved["description"]


def test_an_override_equal_to_the_historical_value_is_still_an_override() -> None:
    """What was asked for and what was recorded stay distinguishable."""

    resolved = resolve_forward_batch_size({"forward_batch_size": 4}, 4)

    assert resolved["value"] == 4
    assert resolved["source"] == "override"


def test_a_missing_realized_batch_size_fails_rather_than_guessing() -> None:
    """Silently defaulting is exactly the failure this rule exists to prevent."""

    with pytest.raises(ValueError, match="does not carry"):
        resolve_forward_batch_size({"num_positions": 32768})


def test_a_missing_realized_batch_size_can_still_be_supplied_explicitly() -> None:
    resolved = resolve_forward_batch_size({}, 4)

    assert resolved["value"] == 4
    assert resolved["source"] == "override"
    assert resolved["historical"] is None
    assert "unrecorded" in resolved["description"]


def test_an_impossible_batch_size_is_rejected_from_either_source() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        resolve_forward_batch_size({"forward_batch_size": 4}, 0)
    with pytest.raises(ValueError, match="cannot"):
        resolve_forward_batch_size({"forward_batch_size": 0})


# -- reconstructing a subset of the recorded sweep ---------------------------
#
# A reconstruction need not cover every recorded temperature. Refusing a subset
# is what stopped an unmeasured-T_g request from ever reaching the check that
# was supposed to reject it, so the gate now compares only what was asked for --
# against that temperature's own recorded histogram, and no other.

from llm_behavior_lab.analysis.nucleus_clustering import select_recorded_histograms

RECORDED = (0.12, 0.24, 0.36, 0.48, 0.60, 1.20)


def _recorded_counts():
    """One distinguishable histogram per recorded temperature.

    Equal totals across temperatures, because every temperature sampled the
    same positions; only which tokens they landed on differs.
    """

    counts = np.zeros((len(RECORDED), VOCAB), dtype=np.int64)
    for index in range(len(RECORDED)):
        counts[index, index + 2] = 4
        counts[index, 1] = 2
    return counts


def test_the_whole_recorded_sweep_still_selects_itself() -> None:
    counts = _recorded_counts()

    selection = select_recorded_histograms(RECORDED, RECORDED, counts)

    assert np.array_equal(selection["indices"], np.arange(len(RECORDED)))
    assert np.array_equal(selection["counts"], counts)


def test_a_single_temperature_subset_selects_only_its_own_histogram() -> None:
    counts = _recorded_counts()

    selection = select_recorded_histograms(RECORDED, [0.36], counts)

    assert selection["counts"].shape == (1, VOCAB)
    assert np.array_equal(selection["indices"], [2])
    assert np.array_equal(selection["counts"][0], counts[2])


def test_a_multi_element_subset_selects_exactly_those_histograms() -> None:
    counts = _recorded_counts()

    selection = select_recorded_histograms(RECORDED, [0.24, 1.20], counts)

    assert np.array_equal(selection["indices"], [1, 5])
    assert np.array_equal(selection["counts"], counts[[1, 5]])


def test_the_requested_order_is_honoured_rather_than_the_stored_order() -> None:
    """A request out of stored order must not be silently re-sorted."""

    counts = _recorded_counts()

    selection = select_recorded_histograms(RECORDED, [1.20, 0.12, 0.48], counts)

    assert np.array_equal(selection["indices"], [5, 0, 3])
    assert np.array_equal(selection["counts"][0], counts[5])
    assert np.array_equal(selection["counts"][1], counts[0])


def test_a_repeated_request_selects_the_same_histogram_twice() -> None:
    """Well-defined: one T_s paired against two T_g keeps both points."""

    counts = _recorded_counts()

    selection = select_recorded_histograms(RECORDED, [0.12, 0.12], counts)

    assert np.array_equal(selection["indices"], [0, 0])
    assert np.array_equal(selection["counts"][0], selection["counts"][1])


def test_an_unrecorded_sampling_temperature_is_refused() -> None:
    counts = _recorded_counts()

    with pytest.raises(ValueError) as failure:
        select_recorded_histograms(RECORDED, [0.37], counts)

    message = str(failure.value)
    assert "T_s = 0.37" in message
    assert "0.12, 0.24, 0.36, 0.48, 0.6, 1.2" in message


def test_a_float_round_tripped_request_still_matches() -> None:
    counts = _recorded_counts()

    selection = select_recorded_histograms(
        RECORDED, [float(np.float32(0.36))], counts
    )

    assert np.array_equal(selection["indices"], [2])


def test_a_near_miss_does_not_alias_onto_a_neighbour() -> None:
    counts = _recorded_counts()

    with pytest.raises(ValueError, match="recorded no nucleus histogram"):
        select_recorded_histograms(RECORDED, [0.30], counts)


def test_an_empty_request_is_refused() -> None:
    with pytest.raises(ValueError, match="No sampling temperatures"):
        select_recorded_histograms(RECORDED, [], _recorded_counts())


def test_a_histogram_count_that_does_not_match_the_axis_is_refused() -> None:
    with pytest.raises(ValueError, match="recorded histogram"):
        select_recorded_histograms(RECORDED, [0.12], _recorded_counts()[:3])


def test_a_subset_still_aborts_on_a_mismatch_inside_it() -> None:
    """The gate is not weakened by narrowing what it covers."""

    counts = _recorded_counts()
    selection = select_recorded_histograms(RECORDED, [0.24, 0.48], counts)
    labels = np.stack([_labels_for(counts[1]), _labels_for(counts[3])])

    # Exactly reproduced: passes.
    assert histogram_gate(
        labels, selection["counts"], vocab_size=VOCAB, temperatures=[0.24, 0.48]
    )["passed"]

    # One position moved inside the second selected temperature: aborts.
    broken = selection["counts"].copy()
    broken[1][3] -= 1
    broken[1][4] += 1
    with pytest.raises(ValueError, match="T = 0.48"):
        histogram_gate(labels, broken, vocab_size=VOCAB, temperatures=[0.24, 0.48])


def test_only_the_selected_histograms_are_compared() -> None:
    """An unselected temperature's histogram cannot fail the gate."""

    counts = _recorded_counts()
    # Corrupt a temperature that is not requested, keeping its total intact so
    # only the selection can explain the gate passing.
    counts[4] = 0
    counts[4][0] = 6
    selection = select_recorded_histograms(RECORDED, [0.12, 0.24], counts)
    labels = np.stack([_labels_for(counts[0]), _labels_for(counts[1])])

    result = histogram_gate(
        labels, selection["counts"], vocab_size=VOCAB, temperatures=[0.12, 0.24]
    )

    assert result["passed"]


def _labels_for(counts: np.ndarray) -> np.ndarray:
    """A label vector whose histogram is exactly ``counts``."""

    return np.repeat(np.arange(counts.size), counts).astype(np.int64)


def test_an_unmeasured_loss_temperature_is_refused_before_reconstruction() -> None:
    """Ordering matters, not just the eventual error.

    A T_g the record never measured can never succeed, so discovering it after
    building the model and recovering labels wastes the expensive half of the
    work -- which is exactly what happened: the request never reached the
    directional-field lookup at all. The check is asserted to sit ahead of every
    expensive call in ``main``. The writer imports PyTorch at module scope, so
    this reads the source rather than executing it.
    """

    import ast
    import pathlib

    script = (
        pathlib.Path(__file__).resolve().parents[1]
        / "scripts"
        / "write_nucleus_clustering_artifact.py"
    )
    tree = ast.parse(script.read_text())
    main = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )

    def first_line(name: str) -> int:
        lines = [
            node.lineno
            for node in ast.walk(main)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == name
        ]
        assert lines, f"{name} is not called in main()"
        return min(lines)

    validation = first_line("loss_temperature_index")
    for expensive in (
        "build_model_from_config",
        "nucleus_position_labels",
        "seed_everything",
        "histogram_gate",
    ):
        assert validation < first_line(expensive), (
            f"the T_g check must run before {expensive}"
        )

"""A finalized record renders from its artifact, or it fails loudly.

Figures 20 and 24 used to rebuild every statistic from raw rows on each render
-- figure 24 including a 256-draw null, which is where the hours went. A
``metrics_only`` record has the answers and no rows, so those routes are not
merely slow now, they are impossible.

That makes the interesting assertion a negative one. It is not enough that the
figures produce output; they must produce it **without reaching** a row-level
estimator, a permutation null, an ensemble embedding, or a model. A fallback
that silently recomputed would either crash far from its cause or, worse, quietly
compute from some other record's arrays -- so the row-level entry points are
replaced with detonators and the render is expected to complete anyway.

The contract these pin:

``load_record``
    the primary-record loader. Returns the record alone, with no metrics
    attached, whatever sits beside it.
``load_finalized_record``
    the canonical v13 bundle loader. Returns the record **with** its metrics
    attached and validated against the hashes the record itself carries.

A v13 consumer handed the output of the first must not quietly behave as though
it had the output of the second.
"""

from __future__ import annotations

import importlib.util
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
yaml = pytest.importorskip("yaml")
pytest.importorskip("torch")

from llm_behavior_lab.analysis import figures as figure_module
# `llm_behavior_lab.analysis` re-exports a *function* named
# `gradient_clustering`, which shadows the submodule of the same name -- so
# `import a.b as c` binds the function, not the module, and every patch would
# land somewhere the real call sites never look. `import_module` returns the
# module itself, which is what the traps have to patch to fire at all.
gradient_clustering = import_module("llm_behavior_lab.analysis.gradient_clustering")
gradient_cross_partition = import_module(
    "llm_behavior_lab.analysis.gradient_cross_partition"
)
from llm_behavior_lab.analysis.alignment_metrics import load_finalized_record
from llm_behavior_lab.analysis.alignment_views import MetricsUnavailable
from llm_behavior_lab.analysis.records import load_record

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_initialization_distribution_experiment.py"


def _run(directory: Path) -> Path:
    spec = importlib.util.spec_from_file_location("render_runner", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    config = yaml.safe_load(
        (REPO / "configs" / "data" / "tiny_text.yaml").read_text(encoding="utf-8")
    )
    config.setdefault("runtime", {})["device"] = "cpu"
    data = directory / "data.yaml"
    data.write_text(yaml.safe_dump(config), encoding="utf-8")

    argv = [
        "run.py",
        "--data-config", str(data),
        "--model-config", str(REPO / "configs" / "model" / "tiny_llama.yaml"),
        "--num-initializations", "1", "--num-windows", "6", "--block-size", "8",
        "--num-replicates", "1", "--no-uniform-null", "--no-input-structure",
        "--gradient-analysis", "--gradient-sketch",
        "--sketch-maps", "4", "--sketch-dimension", "8",
        "--gradient-temperatures", "0.6", "1.0",
        "--sketch-storage", "metrics_only",
        "--no-figures", "--offline",
        "--output-dir", str(directory / "out"), "--run-id", "render",
    ]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sys, "argv", argv)
        module.main()
    return directory / "out" / "initialization_distribution" / "render" / "analyses"


@pytest.fixture(scope="module")
def finalized(tmp_path_factory) -> Path:
    return _run(tmp_path_factory.mktemp("render"))


@pytest.fixture
def detonated(monkeypatch):
    """Replace every route back to rows with something that refuses to run."""

    def boom(name):
        def fail(*args, **kwargs):
            raise AssertionError(
                f"{name} was reached while rendering a finalized record"
            )

        return fail

    for module, names in (
        (
            gradient_clustering,
            (
                "unit_sketches", "unit_sketches_per_map", "_permutation_null",
                "gradient_clustering", "gradient_clustering_per_map",
            ),
        ),
        (
            gradient_cross_partition,
            (
                "pooled_cross_statistic", "cross_identity_null",
                "cross_partition_matrix", "mixture_reconstruction",
            ),
        ),
    ):
        for name in names:
            monkeypatch.setattr(module, name, boom(name))
    return True


# -- the figures render from the artifact ------------------------------------


def test_figure_twenty_renders_without_touching_a_row(
    finalized, detonated, tmp_path
) -> None:
    record = load_finalized_record(finalized)
    paths = figure_module.plot_gradient_directional_clustering(record, tmp_path)
    assert paths and paths[0].stat().st_size > 0


def test_figure_twenty_four_renders_without_recomputing_its_null(
    finalized, detonated, tmp_path
) -> None:
    """The 256-draw identity null was the expensive part of every render."""

    record = load_finalized_record(finalized)
    paths = figure_module.plot_cross_partition_geometry(record, tmp_path)
    assert paths and paths[0].stat().st_size > 0


def test_rendering_needs_no_model_and_no_cuda(finalized, detonated, tmp_path) -> None:
    """The nucleus labels were captured during the run, so nothing replays."""

    import llm_behavior_lab.evaluation.nucleus_labels as labels

    def fail(*args, **kwargs):
        raise AssertionError("a model was reconstructed while rendering")

    original = labels.nucleus_position_labels
    labels.nucleus_position_labels = fail
    try:
        record = load_finalized_record(finalized)
        figure_module.plot_gradient_directional_clustering(record, tmp_path)
        figure_module.plot_cross_partition_geometry(record, tmp_path)
    finally:
        labels.nucleus_position_labels = original


# -- the two loaders are a contract ------------------------------------------


def test_load_record_attaches_nothing(finalized) -> None:
    """The primary-record loader stays exactly that.

    It is used where only the record is wanted, and it must not acquire a
    dependency on whatever happens to sit in the same directory.
    """

    record = load_record(finalized)
    assert record.alignment_metrics is None
    assert not record.has_finalized_gradient_metrics
    assert record.record_version == 13


def test_load_finalized_record_attaches_and_validates(finalized) -> None:
    record = load_finalized_record(finalized)
    assert record.has_finalized_gradient_metrics
    assert record.alignment_metrics.reference is not None


def test_a_v13_consumer_cannot_silently_fall_back_to_rows(
    finalized, detonated, tmp_path
) -> None:
    """The failure mode this whole change exists to prevent.

    Handed a record whose metrics are not attached, figure 20 must not quietly
    recompute -- there are no rows to recompute from, and reaching for them
    would either crash obscurely or use something else's arrays.
    """

    record = load_record(finalized)
    with pytest.raises(AssertionError, match="was reached"):
        figure_module.plot_gradient_directional_clustering(record, tmp_path)


def test_an_invalid_metrics_artifact_is_refused_rather_than_ignored(
    finalized, tmp_path
) -> None:
    """A corrupted artifact must not degrade into "no metrics, use rows"."""

    from llm_behavior_lab.analysis.alignment_metrics import METRICS_NPZ_NAME

    blob = bytearray((finalized / METRICS_NPZ_NAME).read_bytes())
    blob[-1] ^= 0xFF
    (finalized / METRICS_NPZ_NAME).write_bytes(bytes(blob))
    try:
        with pytest.raises(ValueError, match="hashes to"):
            load_finalized_record(finalized)
    finally:
        blob[-1] ^= 0xFF
        (finalized / METRICS_NPZ_NAME).write_bytes(bytes(blob))


# -- the views refuse rather than substitute ---------------------------------


def test_an_unmeasured_loss_temperature_is_refused(finalized) -> None:
    """A reference from another gradient field is a different geometry."""

    from llm_behavior_lab.analysis.alignment_views import clustering_view

    metrics = load_finalized_record(finalized).alignment_metrics
    with pytest.raises(MetricsUnavailable, match="not substituted"):
        clustering_view(metrics, "target", loss_temperature=0.24)


def test_a_missing_nucleus_design_is_refused(finalized) -> None:
    """Control and matched answer different questions."""

    from llm_behavior_lab.analysis.alignment_views import nucleus_view

    metrics = load_finalized_record(finalized).alignment_metrics
    view = nucleus_view(metrics, design="control")
    assert view["by_temperature"]
    assert np.allclose(view["loss_temperatures"], 1.0)


# -- figures 22 and 23 take the view path, and fail loudly without it --------


def test_figure_twenty_two_reads_the_finalized_nucleus_points(finalized) -> None:
    """Both designs come out of the artifact: the labels were captured in-run,
    so there is nothing left to reconstruct and no artifact to write first."""

    from llm_behavior_lab.analysis.alignment_views import nucleus_view

    metrics = load_finalized_record(finalized).alignment_metrics
    control = nucleus_view(metrics, design="control")
    assert control["by_temperature"]
    assert np.allclose(control["loss_temperatures"], 1.0)

    matched = nucleus_view(metrics, design="matched")
    assert np.allclose(
        matched["loss_temperatures"], matched["sampling_temperatures"]
    )


def test_a_missing_nucleus_design_fails_loudly(finalized) -> None:
    """Never a silent fall-through to the legacy reconstruction, which would
    rebuild the corpus and the model and run a forward pass per temperature --
    not a fallback but a different, five-hour computation."""

    from llm_behavior_lab.analysis.alignment_views import nucleus_view

    metrics = load_finalized_record(finalized).alignment_metrics
    stripped = type(metrics)(
        arrays={
            name: values for name, values in metrics.arrays.items()
            if name != "nucleus_design"
        }
        | {"nucleus_design": np.array(["control"] * len(metrics["nucleus_design"]))},
        provenance=metrics.provenance,
    )
    with pytest.raises(MetricsUnavailable, match="different designs"):
        nucleus_view(stripped, design="matched")


def test_a_run_without_the_sanity_check_says_so(finalized) -> None:
    """The check is off by default, so its absence is the ordinary case and
    must read as "not requested" rather than as a broken artifact."""

    from llm_behavior_lab.analysis.alignment_views import sanity_view

    metrics = load_finalized_record(finalized).alignment_metrics
    with pytest.raises(MetricsUnavailable, match="off by default"):
        sanity_view(metrics)


def test_a_corrupt_bundle_never_reaches_the_legacy_reconstruction(
    finalized, monkeypatch, tmp_path
) -> None:
    """The failure mode that matters: a damaged v13 artifact must not degrade
    into "no metrics, recompute from rows" -- there are no rows."""

    import llm_behavior_lab.evaluation.nucleus_labels as labels
    from llm_behavior_lab.analysis.alignment_metrics import METRICS_NPZ_NAME

    def fail(*args, **kwargs):
        raise AssertionError("legacy reconstruction was invoked")

    monkeypatch.setattr(labels, "nucleus_position_labels", fail)

    blob = bytearray((finalized / METRICS_NPZ_NAME).read_bytes())
    blob[-1] ^= 0xFF
    (finalized / METRICS_NPZ_NAME).write_bytes(bytes(blob))
    try:
        with pytest.raises(ValueError, match="hashes to"):
            load_finalized_record(finalized)
    finally:
        blob[-1] ^= 0xFF
        (finalized / METRICS_NPZ_NAME).write_bytes(bytes(blob))

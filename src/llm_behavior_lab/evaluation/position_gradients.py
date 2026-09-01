"""Exact per-position parameter-gradient norms at a fixed initialization.

For one evaluation position ``d`` whose true next token is ``y_d``:

.. code-block:: text

    z_d       = model logits at position d, restricted to the tokenizer vocabulary
    z_d^elig  = z_d with non-eligible (structural) IDs driven to -inf
    ell_d     = -log softmax(z_d^elig)[y_d]     single position, eligible support
    g_d       = || grad_theta ell_d ||_2        theta = every requires_grad parameter
    greedy_d  = argmax z_d^elig                 same masked logits, same forward

``g_d`` is the **exact** global L2 norm over every trainable parameter. It is not
a logit gradient, not a hidden-state gradient, not an LM-head-only gradient, and
not the gradient of a window-averaged loss. No cheaper proxy is substituted
anywhere in this module.

**Not to be confused with** :mod:`llm_behavior_lab.evaluation.gradient_norms`,
which is a different Phase 5 diagnostic: it measures squared norms of gradients
with respect to *decoder block output activations*, via forward hooks, taken from
the *window-averaged* loss. That module answers "are gradients stable through
depth"; this one answers "how hard does one position's true-token loss pull on
the parameters".

Two properties make the measurement affordable and safe.

*Affordable*: position ``d``'s logits depend only on tokens at or before ``d`` in
its own window, so one forward pass per window serves every position in that
window; the graph is retained and re-differentiated once per position. Nothing
shaped ``[num_positions, num_parameters]`` is ever materialized -- at ``D=32768``
over 8.6M parameters that would be about 1.1 TB in float32. Live at any instant are one
window's graph and one gradient set; only ``[D_g]``-sized vectors survive.

*Safe*: gradients are taken with :func:`torch.autograd.grad`, which returns them
instead of accumulating into ``parameter.grad``. There is no optimizer, no
``zero_grad``, no clipping, and no in-place weight operation, so the model's
parameters, buffers, and existing ``.grad`` state are left exactly as found. The
entry ``training``/``eval`` mode is captured and restored even if the measurement
raises.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

from llm_behavior_lab.evaluation.guessing import (
    apply_support_mask,
    eligible_support_mask,
    greedy_guess_ids,
    restrict_to_support,
)
from llm_behavior_lab.evaluation.init_distribution import EvaluationPositions

__all__ = [
    "CANONICAL_GRADIENT_TEMPERATURE",
    "GRADIENT_DEFINITION",
    "GRADIENT_TEMPERATURES",
    "PositionGradientResult",
    "DEFAULT_SANITY_POSITIONS",
    "DEFAULT_SKETCH_DIMENSION",
    "production_sketch_map",
    "production_sketch_tables",
    "compute_position_gradient_norms",
    "evenly_spaced_indices",
    "masked_evaluation_logits",
    "single_position_losses",
]

#: Persisted with every result so a stored number can never be read without the
#: definition that produced it.
GRADIENT_DEFINITION = (
    "l2_norm_of_gradient_of_single_position_next_token_cross_entropy"
    "_with_respect_to_all_trainable_parameters"
)

#: Temperatures at which the loss itself is defined. The six established sweep
#: values plus ``T = 1``, the canonical baseline.
#:
#: Temperature enters the **loss**, not a post-hoc rescaling of a result:
#: ``ell_T(d) = -log softmax(z_d / T)[y_d]``. The gradient therefore changes with
#: ``T`` through two routes at once -- the explicit ``1/T`` in
#: ``d ell_T / d z_i = (p_T(i) - 1[i = y]) / T`` and the sharpening of ``p_T``
#: itself. Both are intended. No compensating ``T^2`` factor is applied; this is
#: plain temperature-scaled cross entropy, not a distillation loss.
#:
#: Greedy identity is untouched: softmax is strictly increasing, so
#: ``argmax softmax(z/T) = argmax z`` for every positive ``T``. The guessing bias
#: is held fixed while the learning signal varies, which is the whole design.
GRADIENT_TEMPERATURES = (0.12, 0.24, 0.36, 0.48, 0.60, 1.00, 1.20)

#: The canonical baseline inside that grid.
CANONICAL_GRADIENT_TEMPERATURE = 1.00

#: Width of the per-position gradient sketch. 512 buckets keep the persisted
#: array to about 67 MB at D = 32768 in float32, and a count sketch preserves
#: inner products with a relative error of order ``1/sqrt(K)`` -- roughly 4% here
#: -- which is fine for comparing group means but not for trusting any single
#: pairwise cosine. Validated rather than assumed: see the sketch tests.
#:
#: Runtime cost is **not yet measured**. Each position adds one scatter-add over
#: every parameter on top of a backward pass that already dominates it, so the
#: overhead is expected to be small relative to the roughly 2027 s the gradient
#: analysis took at D = 32768 -- but that expectation stands until the matched
#: baseline and sketch smokes are timed on the cluster.
DEFAULT_SKETCH_DIMENSION = 512

#: Device-resident representation of one CountSketch map. The map is *drawn* as
#: int64 buckets and float64 signs -- that is its public form, and narrowing the
#: draw would move the RNG stream -- and narrowed only on the way to the device,
#: where one map costs 16 bytes per parameter otherwise. At 134M parameters that
#: is 2 GiB of a single card per map.
#:
#: ``int32`` because ``index_add_`` accepts int32 or int64 and never int16, and
#: ``K`` does not approach 2^31. ``int8`` because signs are +/-1 and promote to
#: exactly +/-1.0 at the multiply.
_SKETCH_BUCKET_DEVICE_DTYPE = torch.int32
_SKETCH_SIGN_DEVICE_DTYPE = torch.int8

#: Device bytes per parameter per map, computed from the dtypes above rather than
#: written down. A previous hard-coded copy of this number in the benchmark went
#: stale the moment the dtypes changed; deriving it means it cannot. Private:
#: the device representation is an implementation choice, not a contract.
_SKETCH_MAP_DEVICE_BYTES_PER_PARAMETER = (
    torch.empty(0, dtype=_SKETCH_BUCKET_DEVICE_DTYPE).element_size()
    + torch.empty(0, dtype=_SKETCH_SIGN_DEVICE_DTYPE).element_size()
)

#: Positions whose complete gradient is retained for the fidelity sanity check.
#: Twelve gives 66 unique pairs against eight's 28, which is a far more
#: informative fidelity comparison, at about 394 MiB of temporary CPU float32 --
#: modest, held only until the offline analysis finishes, and never on the GPU.
DEFAULT_SANITY_POSITIONS = 12


@dataclass(frozen=True)
class PositionGradientResult:
    """Per-position gradient norms for one model initialization.

    Every vector has length ``D_g``, the number of evaluated positions, and the
    four are aligned entry by entry. ``position_indices`` is the flat index
    ``window * block_size + offset`` into the experiment's evaluation grid, so a
    subset of windows stays traceable back to the positions it came from.

    ``losses`` is carried for validation and for the console sanity check (at
    initialization the mean should sit near ``log`` of the eligible support). It
    is deliberately *not* persisted: the record schema stores the observables the
    analysis is defined on, and this one is recoverable from a rerun.
    """

    position_indices: torch.Tensor
    target_ids: torch.Tensor
    greedy_ids: torch.Tensor
    #: ``[N_T, D_g]`` exact norms, one row per temperature.
    temperature_gradient_norms: torch.Tensor
    #: ``[N_T, D_g]`` the losses those gradients were taken of.
    temperature_losses: torch.Tensor
    #: The temperature grid, in the order the rows are stored.
    temperatures: tuple[float, ...]
    #: ``[D_g]`` canonical ``T = 1`` slice, the established observable.
    gradient_norms: torch.Tensor
    losses: torch.Tensor
    parameter_count: int
    num_parameter_tensors: int
    num_windows: int
    block_size: int
    seconds: float
    definition: str = GRADIENT_DEFINITION
    softmax_support: str = "eligible"
    #: Scalars of the canonical-temperature correct-vs-wrong vector split, or
    #: ``None`` when the diagnostic was not requested.
    vector_split: dict[str, Any] | None = None
    #: ``[D_g, K]`` at ``M = 1`` and ``[D_g, M, K]`` above it: the count sketch of
    #: each position's canonical gradient, or ``None`` when sketching was not
    #: requested. Always the canonical row of ``temperature_gradient_sketches``
    #: rather than a separately measured quantity, so the two cannot disagree.
    #:
    #: The rank is conditional; **the map count is not read from it**. It is
    #: ``sketch_protocol["map_count"]``, which is written at every ``M`` including
    #: one. Inferring ``M`` from rank or width would misread any future array that
    #: happens to gain an axis for another reason.
    gradient_sketches: torch.Tensor | None = None

    #: ``[N_T, D_g, K]`` at ``M = 1``, ``[N_T, D_g, M, K]`` above it: the count
    #: sketch of every position's gradient at every measured loss temperature, or
    #: ``None`` when sketching was not requested.
    #:
    #: Direction, unlike magnitude, is not recoverable from a scalar, so a
    #: directional analysis at ``T_g != 1`` needs its own projection. Each row is
    #: produced from the very gradient set whose exact norm sits in the matching
    #: row of ``temperature_gradient_norms`` -- the same backward pass, no second
    #: traversal -- which is what keeps the pair aligned by construction.
    #:
    #: Stored float32. The projection itself accumulates in float64 inside
    #: :meth:`_GradientSketcher.project`; only the finished ``[K]`` vector is
    #: narrowed, and the record persists sketches at float32 anyway, so the
    #: canonical row is bit-for-bit what it always was. At ``N_T = 7,
    #: D_g = 32768, K = 512`` this is about 470 MB, against 940 MB in float64.
    temperature_gradient_sketches: torch.Tensor | None = None
    #: Provenance of that projection, or ``None``.
    sketch_protocol: dict[str, Any] | None = None
    #: ``[m, P]`` complete float32 gradients for the sanity subset, or ``None``.
    #: Temporary: the caller runs the offline fidelity analysis and drops them.
    exact_gradients: torch.Tensor | None = None
    #: Flat indices of those positions, or ``None``.
    exact_positions: torch.Tensor | None = None
    #: ``[m, M, K]`` float32 sketches of the same positions, taken from the
    #: **live** production projection rather than re-derived afterwards.
    #:
    #: Rebuilding the maps to re-project would compare exact gradients against a
    #: reconstruction of the instrument, not against the instrument. Temporary:
    #: the caller computes the fidelity summary and releases these with the
    #: gradients.
    exact_sketches: torch.Tensor | None = None
    #: The production sketch map as ``(buckets, signs)`` NumPy vectors, so the
    #: offline fidelity analysis projects with the map the run actually used
    #: rather than re-deriving one from a different RNG.
    #:
    #: **Map 0, always**, in its historical ``int64``/``float64`` form, whatever
    #: ``M`` was. No all-map table export exists: the ordered ``map_seeds`` in
    #: ``sketch_protocol`` plus the deterministic construction identify the whole
    #: bank, and exporting ``M`` full-length table pairs would add ``M`` times the
    #: largest arrays in the result for information already recorded.
    sketch_map: tuple[Any, Any] | None = None
    #: Parameter element counts in order, so an offline analysis can rebuild
    #: another realization of the same production construction.
    sketch_tensor_sizes: list[int] | None = None

    @property
    def canonical_index(self) -> int:
        """Row holding the canonical ``T = 1`` observable."""

        return self.temperatures.index(CANONICAL_GRADIENT_TEMPERATURE)

    @property
    def num_positions(self) -> int:
        """Number of evaluated positions, ``D_g``."""

        return int(self.gradient_norms.shape[0])

    def as_metadata(self, **extra: Any) -> dict[str, Any]:
        """Return a JSON-serializable description of what was measured."""

        payload: dict[str, Any] = {
            "definition": self.definition,
            "softmax_support": self.softmax_support,
            "includes_all_trainable_parameters": True,
            "parameter_count": self.parameter_count,
            "num_parameter_tensors": self.num_parameter_tensors,
            "num_positions": self.num_positions,
            "num_windows": self.num_windows,
            "block_size": self.block_size,
            "temperatures": list(self.temperatures),
            "canonical_temperature": CANONICAL_GRADIENT_TEMPERATURE,
            "temperature_enters_the_loss": True,
            "compensating_t_squared_factor": False,
            "mean_loss": float(self.losses.mean().item()),
            "mean_gradient_norm": float(self.gradient_norms.mean().item()),
            "seconds": round(self.seconds, 3),
        }
        if self.vector_split is not None:
            payload["vector_split"] = dict(self.vector_split)
        if self.sketch_protocol is not None:
            payload["gradient_sketch"] = dict(self.sketch_protocol)
        payload.update(extra)
        return payload


def production_sketch_tables(
    tensor_sizes: Sequence[int], dimension: int, seed: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The production CountSketch map, as one (buckets, signs) pair per tensor.

    This is the single definition of what "the production map" means. Both the
    live sketcher and every offline robustness analysis go through it, so the
    two can no longer drift apart -- which they did once already, when an offline
    helper rebuilt the map with a different RNG and produced a structurally
    unrelated projection.

    A generator per parameter tensor, seeded ``seed + 1000003 * index``, drawing
    buckets and then signs. The per-tensor seeding is what makes the map
    independent of how many tensors precede a given one, so adding a buffer
    somewhere else in the model cannot silently reshuffle it.

    Torch's generator specifically. Varying only ``seed`` or ``dimension`` gives
    another realization of *this* construction, which is the question a
    robustness analysis is asking; a NumPy generator on the same seed would
    answer a different one.
    """

    tables = []
    for index, count in enumerate(tensor_sizes):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 1000003 * index)
        buckets = torch.randint(
            0, int(dimension), (int(count),), generator=generator, dtype=torch.long
        )
        signs = torch.randint(
            0, 2, (int(count),), generator=generator, dtype=torch.int8
        )
        tables.append((buckets, signs.double() * 2.0 - 1.0))
    return tables


#: Stride between the base seeds of consecutive production CountSketch maps.
#:
#: Large and coprime to the per-tensor stride ``1000003`` inside
#: :func:`production_sketch_tables`. Two ``(map, tensor)`` pairs collide only when
#: ``1_000_000_007 * (m1 - m2) == 1000003 * (i2 - i1)``; both strides are prime and
#: distinct, so that forces ``1000003`` to divide ``m1 - m2`` **and** ``1_000_000_007``
#: to divide ``i2 - i1``. The smallest non-trivial solution is therefore a
#: 1,000,003-map gap alongside a 1,000,000,007-tensor gap.
#:
#: That is a bound on the realistic envelope, **not** a proof that no collision
#: exists for arbitrarily large indices. Campaigns run a handful of maps over
#: hundreds of tensors, and the schedule is collision-free throughout the tested
#: range -- asserted executably over 8 maps x 2,000 tensors in
#: ``tests/test_countsketch_replicas.py``. Two maps sharing a tensor seed would
#: share that tensor's buckets and signs, which would make them correlated exactly
#: where the replica design assumes independence.
_SKETCH_MAP_SEED_STRIDE = 1_000_000_007

#: Bounds ``torch.Generator.manual_seed`` accepts. Measured, not assumed: outside
#: this range it raises ``ValueError: Overflow when unpacking long long``. The
#: derived per-tensor seeds are checked against it so a large ``base_seed`` fails
#: with an explanation instead of overflowing into an unrelated stream.
_MIN_GENERATOR_SEED = -(2**63)
_MAX_GENERATOR_SEED = 2**64 - 1


#: Identifies how ``map_seeds`` were derived, so a stored bank can be checked
#: rather than trusted. The rule name says which arithmetic produced them; the
#: version is bumped only if that arithmetic ever changes, which would make every
#: map but the zeroth a different projection.
_SKETCH_SEED_DERIVATION = "base_seed + 1000000007 * map_index"
_SKETCH_SEED_DERIVATION_VERSION = 1

#: Schema version of the emitted ``sketch_protocol``. Deliberately independent of
#: the record schema: the two move for different reasons. Kept as a literal here
#: rather than imported from :mod:`llm_behavior_lab.analysis.records`, because
#: that module is NumPy-only by design and this one is the torch side of the
#: boundary; the record layer validates the value it finds. A test asserts the
#: two constants agree.
_SKETCH_PROTOCOL_SCHEMA_VERSION = 2


def _validated_map_count(map_count: Any, *, name: str) -> int:
    """Return ``map_count`` as a positive ``int``, or explain why it is not.

    One definition shared by :class:`_GradientSketcher` and
    :func:`compute_position_gradient_norms`, so the two cannot drift into
    disagreeing about what a legal map count is -- which would let a value the
    measurement accepts reach a sketcher that rejects it, or worse, the reverse.

    ``bool`` is refused explicitly: ``True == 1`` in Python, so ``map_count=True``
    would otherwise pass as a single map and read as if someone had asked a
    yes/no question and got a count.

    Args:
        map_count: The candidate value.
        name: The caller's parameter name, so the message names what the caller
            actually passed rather than an internal spelling.

    Returns:
        The value as an ``int``.

    Raises:
        ValueError: If it is boolean, non-integral, or below one.
    """

    try:
        count = int(map_count)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be an integer of at least one; got {map_count!r}."
        ) from None
    if isinstance(map_count, bool) or count != map_count or count < 1:
        raise ValueError(
            f"{name} must be an integer of at least one; got {map_count!r}."
        )
    return count


def _production_sketch_map_seed(base_seed: int, map_index: int) -> int:
    """Base seed of production map ``map_index``.

    ``map_seed(base, m) = base + 1_000_000_007 * m``, so **map 0 is exactly the
    supplied base seed**. That is the whole compatibility story: at ``m = 0`` the
    per-tensor seeds reduce to the historical ``base + 1000003 * i``, the draw
    order is untouched, and the map is bit-identical to every map this project has
    ever produced.

    Private on purpose. The device representation and the replica seed schedule
    are implementation choices, not contracts, and neither belongs in ``__all__``.

    Args:
        base_seed: The production seed the run was configured with.
        map_index: Zero-based replica index.

    Returns:
        The base seed for that map's per-tensor generators.

    Raises:
        ValueError: If ``map_index`` is negative, or the derived seed falls
            outside the range ``torch.Generator`` accepts.
    """

    if int(map_index) < 0:
        raise ValueError(f"map_index must be non-negative; got {map_index}.")
    seed = int(base_seed) + _SKETCH_MAP_SEED_STRIDE * int(map_index)
    if not _MIN_GENERATOR_SEED <= seed <= _MAX_GENERATOR_SEED:
        raise ValueError(
            f"Map {map_index} derives seed {seed} from base seed {base_seed}, "
            f"which is outside the range torch.Generator accepts "
            f"[{_MIN_GENERATOR_SEED}, {_MAX_GENERATOR_SEED}]. Choose a base seed "
            "that leaves room for every replica rather than letting the "
            "derivation overflow into an unrelated stream."
        )
    return seed


def production_sketch_map(
    tensor_sizes: Sequence[int], dimension: int, seed: int
) -> tuple[Any, Any]:
    """The production map flattened into NumPy bucket and sign vectors.

    Concatenated in parameter order, matching how a captured gradient is
    flattened, so an offline projection lines up entry for entry with the
    sketches the experiment produced.
    """

    import numpy as np

    tables = production_sketch_tables(tensor_sizes, dimension, seed)
    return (
        np.concatenate([buckets.numpy() for buckets, _ in tables]),
        np.concatenate([signs.numpy() for _, signs in tables]),
    )


class _GradientSketcher:
    """Deterministic count sketch of a full parameter gradient.

    The whole point is that ``g_d`` cannot be kept. At 8.6M parameters and 32768
    positions the exact matrix is about 1.1 TB in float32, so directional structure has to
    be measured through a projection that is cheap to apply and preserves the
    only thing being asked about: inner products between gradients.

    A count sketch does exactly that. Every parameter coordinate is assigned,
    once, a bucket ``h(i) in [0, K)`` and a sign ``s(i) in {-1, +1}``, and the
    sketch is ``sketch[h(i)] += s(i) * g[i]``. The signs make the estimator
    unbiased: for any two gradients ``E<sketch(a), sketch(b)> = <a, b>``, with a
    variance that falls as ``1/K``. Cosines computed from sketches are therefore
    unbiased in the inner product and accurate enough for class-level means,
    which is what the clustering analysis consumes.

    Why not a dense Gaussian projection: it would need a ``[P, K]`` matrix, about
    17 GB at ``P = 8.6M, K = 512``. The sketch needs two ``[P]`` tables instead
    and applies in one scatter-add.

    Determinism is structural. Buckets and signs come from a
    :class:`torch.Generator` seeded once from ``seed``, drawn per parameter
    tensor in a fixed order, and never touched again -- so two runs of the same
    experiment produce bitwise the same projection. It draws from its own
    generator and never from the global RNG, so enabling the sketch cannot shift
    any sampling stream in the experiment around it.

    ``map_count`` holds several **independent** maps at once. One map gives an
    unbiased estimate with no handle on its own error; several give a spread to
    read that error off. They are seeded by
    :func:`_production_sketch_map_seed`, and map 0 is exactly the historical map,
    so adding replicas cannot move the established observable. Every map projects
    the same gradient tuple in :meth:`project_maps` -- no gradient is recomputed
    and nothing here ever calls ``autograd``.

    These are **production** replicas. They are not the offline alternate-map
    bank in :mod:`llm_behavior_lab.analysis.countsketch_fidelity`, which
    re-projects retained exact gradients to ask how much a reported error would
    move under a different draw. The two answer different questions and must not
    be conflated.
    """

    def __init__(
        self,
        parameters: Sequence[torch.Tensor],
        *,
        dimension: int = DEFAULT_SKETCH_DIMENSION,
        seed: int = 20240917,
        map_count: int = 1,
    ) -> None:
        if dimension < 1:
            raise ValueError(f"sketch dimension must be positive; got {dimension}.")
        count = _validated_map_count(map_count, name="map_count")
        # Materialized before anything consumes it. The annotation says
        # Sequence, but a caller handing over a generator would otherwise have it
        # exhausted by tensor_sizes below and leave every table list EMPTY --
        # silently, because zip over an empty list simply yields nothing and
        # project() would then return an all-zero sketch with no error at all.
        parameters = tuple(parameters)
        self.dimension = int(dimension)
        #: The **base** seed, unchanged in meaning: it is map 0's seed.
        self.seed = int(seed)
        self.map_count = count
        self.tensor_sizes = [int(parameter.numel()) for parameter in parameters]
        self._map_seeds = tuple(
            _production_sketch_map_seed(self.seed, index) for index in range(count)
        )
        # The map seeds are in range; the per-tensor seeds derived from them must
        # be too, and they run 1000003 higher per tensor. Checked once here, where
        # the tensor count is known, rather than discovered as an overflow deep
        # inside the last map's construction.
        if self.tensor_sizes:
            furthest = self._map_seeds[-1] + 1000003 * (len(self.tensor_sizes) - 1)
            if not _MIN_GENERATOR_SEED <= furthest <= _MAX_GENERATOR_SEED:
                raise ValueError(
                    f"The per-tensor seed for the last tensor of map {count - 1} "
                    f"would be {furthest}, outside the range torch.Generator "
                    f"accepts [{_MIN_GENERATOR_SEED}, {_MAX_GENERATOR_SEED}]."
                )

        # Compact on the device, historical on the host. Each map's tables are
        # drawn exactly as they always were -- int64 buckets, float64 signs, one
        # generator per tensor, buckets before signs -- and only then narrowed,
        # so the RNG stream and every drawn value are untouched.
        #
        # A single fused transfer-and-cast, deliberately: `.to(device).to(dtype)`
        # would allocate the wide table on the device first and leave its block
        # in the caching allocator, which is exactly the memory this exists to
        # avoid. Buckets are int32 because `index_add_` requires int32 or int64
        # and K never approaches 2^31; signs are int8 because they are +/-1 and
        # promote to float64 exactly at the multiply in `project`.
        #
        # One map at a time, and the wide host tables are dropped before the next
        # map is drawn. Building all M first would hold M wide maps in host
        # memory at once -- 16 bytes per parameter each, so 8.6 GiB at M = 4 over
        # 134M parameters -- for no reason, since each is consumed immediately.
        self._map_buckets: list[list[torch.Tensor]] = []
        self._map_signs: list[list[torch.Tensor]] = []
        for map_seed in self._map_seeds:
            tables = production_sketch_tables(
                self.tensor_sizes, self.dimension, map_seed
            )
            self._map_buckets.append(
                [
                    buckets.to(
                        device=parameter.device, dtype=_SKETCH_BUCKET_DEVICE_DTYPE
                    )
                    for (buckets, _), parameter in zip(tables, parameters)
                ]
            )
            self._map_signs.append(
                [
                    signs.to(device=parameter.device, dtype=_SKETCH_SIGN_DEVICE_DTYPE)
                    for (_, signs), parameter in zip(tables, parameters)
                ]
            )
            del tables

        # Map 0's tables *are* the historical attributes -- the same list objects,
        # not copies. Every existing caller and test reads `buckets` and `signs`
        # as flat per-tensor lists, and they still are; the replica bank is the
        # private structure beside them.
        self.buckets = self._map_buckets[0]
        self.signs = self._map_signs[0]

    def numpy_map(self) -> tuple["Any", "Any"]:
        """Export this exact map as flat NumPy bucket and sign vectors.

        The offline fidelity analysis must project with the **same** map the
        experiment used, and it cannot re-derive it: this construction draws from
        ``torch.Generator``, and a NumPy generator seeded identically produces a
        completely different map. Reproducing a Mersenne Twister stream in NumPy
        is not a reasonable thing to maintain, so the map itself is handed over
        instead. One definition, one source of truth.

        Concatenated in parameter order, which is the order a captured gradient
        is flattened in, so the two line up entry for entry.

        Widened back to the historical ``int64`` / ``float64`` on the way out.
        The device tables are stored compactly, but this is the map's public
        form: the fidelity analysis and every persisted ``sketch_map`` expect
        those dtypes. Both widenings are exact -- int32 indices and +/-1 signs
        lose nothing -- so the exported map is identical to what a wide-table
        sketcher would have produced.
        """

        import numpy as np

        return (
            np.concatenate([bucket.cpu().to(torch.int64).numpy() for bucket in self.buckets]),
            np.concatenate([sign.cpu().to(torch.float64).numpy() for sign in self.signs]),
        )

    def _project_tables(
        self,
        grads: Sequence[torch.Tensor],
        buckets: Sequence[torch.Tensor],
        signs: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        """Sketch one gradient set through one map into ``[dimension]`` float64.

        The historical projection body, unchanged, with the map it reads made an
        argument. Extracting it is what lets ``project`` stay exactly the map-0
        path while ``project_maps`` reuses the same arithmetic per map rather
        than reimplementing it.
        """

        sketch = torch.zeros(
            self.dimension, dtype=torch.float64, device=signs[0].device
        )
        for gradient, bucket, sign in zip(grads, buckets, signs):
            sketch.index_add_(0, bucket, gradient.detach().double().reshape(-1) * sign)
        return sketch

    def project(self, grads: Sequence[torch.Tensor]) -> torch.Tensor:
        """Sketch one gradient set into a ``[dimension]`` float64 vector.

        Map 0 only, and unchanged: same shape, same dtype, same device, same
        values as before replicas existed. It is the historical observable, and
        the frozen fixture checks it, so it deliberately does **not** acquire a
        map axis or any stack/squeeze behaviour.
        """

        return self._project_tables(grads, self.buckets, self.signs)

    def project_maps(self, grads: Sequence[torch.Tensor]) -> torch.Tensor:
        """Sketch one gradient set through every map into ``[M, dimension]``.

        The replica estimator's whole premise is that the maps are independent
        projections of **one** gradient, so every row here comes from the same
        materialized ``grads`` -- no recomputation, no second backward pass, and
        no ``autograd`` call anywhere in this class. ``grads`` is materialized
        once because a caller may hand over a one-shot iterable, which the first
        map would otherwise consume, leaving every later map projecting nothing.

        Row 0 is bitwise ``project(grads)``: same map, same helper, same order.

        Returns:
            ``[M, dimension]`` float64 on the parameter device. At ``M = 1`` that
            is ``[1, dimension]`` -- a map axis of length one, never a squeezed
            vector, so the shape says how many maps there were.
        """

        materialized = tuple(grads)
        return torch.stack(
            [
                self._project_tables(materialized, buckets, signs)
                for buckets, signs in zip(self._map_buckets, self._map_signs)
            ]
        )


class _VectorSplitAccumulator:
    """Streaming ``g_correct`` and ``g_wrong`` at the canonical temperature.

    Norm mass answers "how much gradient magnitude does each group generate";
    it cannot answer "where does the first update actually point", because
    ``sum_d ||g_d||`` is not ``||sum_d g_d||`` and the difference is exactly
    however much the individual gradients cancel. That needs the vector sums,
    which is what this accumulates.

    Two parameter-shaped buffers are held, nothing per position: each ``g_d``
    is added into one of them and released with the gradient set that produced
    it. Peak cost is therefore two copies of the parameters -- about 131 MiB at
    8.6M parameters -- and is independent of ``D``.

    float64 throughout, and deliberately with no float32 fallback. The quantity
    is a sum over tens of thousands of terms whose cosine is the thing being
    measured; cancellation is the signal, so accumulating it in the precision
    that loses cancellation would quietly destroy the answer. If the memory ever
    becomes a real problem it should be reported and decided on, not silently
    traded away.

    Only the canonical ``T = 1`` grid row is accumulated: that is the objective
    training will actually use, and one buffer pair per temperature would cost
    seven times the memory for a question nobody asked.
    """

    def __init__(self, parameters: Sequence[torch.Tensor]) -> None:
        self.correct = [
            torch.zeros_like(parameter, dtype=torch.float64) for parameter in parameters
        ]
        self.wrong = [
            torch.zeros_like(parameter, dtype=torch.float64) for parameter in parameters
        ]
        self.num_correct = 0
        self.num_wrong = 0

    def add(self, grads: Sequence[torch.Tensor], *, is_correct: bool) -> None:
        """Accumulate one position's gradient into the matching group."""

        target = self.correct if is_correct else self.wrong
        for buffer, gradient in zip(target, grads):
            buffer.add_(gradient.detach().double())
        if is_correct:
            self.num_correct += 1
        else:
            self.num_wrong += 1

    def summary(self) -> dict[str, Any]:
        """Reduce the two buffers to the reported scalars."""

        squared_correct = torch.zeros((), dtype=torch.float64)
        squared_wrong = torch.zeros((), dtype=torch.float64)
        squared_total = torch.zeros((), dtype=torch.float64)
        dot = torch.zeros((), dtype=torch.float64)
        for correct, wrong in zip(self.correct, self.wrong):
            squared_correct += correct.pow(2).sum().cpu()
            squared_wrong += wrong.pow(2).sum().cpu()
            squared_total += (correct + wrong).pow(2).sum().cpu()
            dot += (correct * wrong).sum().cpu()

        norm_correct = float(squared_correct.sqrt())
        norm_wrong = float(squared_wrong.sqrt())
        product = norm_correct * norm_wrong
        return {
            "temperature": CANONICAL_GRADIENT_TEMPERATURE,
            "norm_correct": norm_correct,
            "norm_wrong": norm_wrong,
            "norm_total": float(squared_total.sqrt()),
            "dot": float(dot),
            # Undefined when either group is empty or its sum vanishes; a cosine
            # of 0.0 would claim orthogonality that was never measured.
            "cosine": float(dot) / product if product > 0.0 else None,
            "num_correct": self.num_correct,
            "num_wrong": self.num_wrong,
        }


def evenly_spaced_indices(total: int, count: int | None = None) -> tuple[int, ...]:
    """Choose ``count`` window indices spread evenly over ``range(total)``.

    Deterministic and seed-free, matching how the evaluation windows themselves
    are chosen: the same subset must be reproducible for every rerun. ``None`` --
    the default everywhere -- means every window, so no reduction of the
    scientific position set happens unless a caller explicitly asks for one.

    Args:
        total: Number of available windows.
        count: Number of windows to keep, or ``None`` for all of them.

    Returns:
        Sorted, distinct window indices.

    Raises:
        ValueError: If ``total`` is not positive, or ``count`` is outside
            ``[1, total]``.
    """

    if total <= 0:
        raise ValueError("total must be positive.")
    if count is None or count == total:
        return tuple(range(total))
    if count <= 0:
        raise ValueError("count must be positive.")
    if count > total:
        raise ValueError(
            f"Requested {count} windows but only {total} are available."
        )
    if count == 1:
        return (0,)
    step = (total - 1) / (count - 1)
    return tuple(int(round(index * step)) for index in range(count))


def _trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Return every parameter the gradient norm is taken over."""

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError(
            "The model exposes no trainable parameters, so a parameter-gradient "
            "norm is undefined."
        )
    return parameters


#: Version tag of the parameter-layout serialization, and the first line of every
#: payload. Bump it only if the encoding below changes; the digest is meaningless
#: without knowing which encoding produced it.
_PARAMETER_LAYOUT_SCHEMA = "parameter_layout/v1"

#: How the layout was enumerated, recorded so a reader need not infer it.
_PARAMETER_ORDERING = (
    "trainable model.named_parameters(), duplicate parameters removed, "
    "module registration order"
)


def _parameter_layout_payload(
    model: torch.nn.Module, parameters: Sequence[torch.Tensor]
) -> str:
    """Serialize the exact ordered parameter domain the sketch is defined over.

    A CountSketch map is a function of the *layout* -- how many scalars there
    are, in which order -- and of nothing else. Two runs whose parameter tuples
    agree here can be compared; two that do not are projecting different domains,
    and a stored map from one is meaningless against the other. The digest of
    this payload is what lets that be checked rather than assumed.

    The encoding, fixed by :data:`_PARAMETER_LAYOUT_SCHEMA`::

        parameter_layout/v1\\n
        {index}\\t{name}\\t{tuple(shape)}\\t{numel}\\n
        ...

    UTF-8, zero-based decimal index, tab-separated, Python integer-tuple shape,
    newline after every entry.

    Deliberately excluded: parameter **values**, the initialization seed, the
    device and the dtype. None of them changes which scalars the sketch runs
    over, so including them would make the digest differ between runs that are in
    fact directly comparable -- exactly the question it exists to answer.

    Args:
        model: The model whose named parameters supply the canonical names.
        parameters: The tuple actually handed to the sketcher.

    Returns:
        The payload string.

    Raises:
        ValueError: If the named enumeration is not the same objects in the same
            order as ``parameters``, or a name contains a tab or newline.
    """

    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    # The names and the sketched tuple are obtained from two different calls, so
    # their agreement is asserted rather than assumed. If they ever diverge --
    # a filtering change on one side, a custom named_parameters override --
    # the digest would silently describe a domain the sketch never used, which
    # is worse than having no digest at all.
    if len(named) != len(parameters):
        raise ValueError(
            f"The named trainable parameters ({len(named)}) do not match the "
            f"sketched parameter tuple ({len(parameters)}); the layout digest "
            "would describe a domain the sketch never projected."
        )
    for index, ((name, named_parameter), sketched) in enumerate(
        zip(named, parameters)
    ):
        if named_parameter is not sketched:
            raise ValueError(
                f"Parameter {index} ({name!r}) from named_parameters() is not the "
                "same object as the one at that position in the sketched tuple, "
                "so the two enumerations disagree on order or identity."
            )

    lines = [f"{_PARAMETER_LAYOUT_SCHEMA}\n"]
    for index, (name, parameter) in enumerate(named):
        if "\t" in name or "\n" in name:
            raise ValueError(
                f"Parameter name {name!r} contains a tab or newline, which are "
                "the field and record separators of "
                f"{_PARAMETER_LAYOUT_SCHEMA}. The encoding would be ambiguous "
                "and two different layouts could hash alike."
            )
        lines.append(
            f"{index}\t{name}\t{tuple(parameter.shape)}\t{parameter.numel()}\n"
        )
    return "".join(lines)


def _parameter_layout_sha256(
    model: torch.nn.Module, parameters: Sequence[torch.Tensor]
) -> str:
    """Lowercase SHA-256 hex digest of :func:`_parameter_layout_payload`."""

    payload = _parameter_layout_payload(model, parameters)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_every_parameter_reached(
    grads: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.nn.Parameter],
) -> None:
    """Fail if a parameter received no gradient at all.

    ``allow_unused=True`` is passed so autograd reports an unreachable parameter
    as ``None`` instead of raising an opaque error. For a decoder-only model
    every parameter lies on the path from any position's logits, so a ``None``
    means the graph is not what this module assumes -- silently treating it as a
    zero contribution would understate the norm without saying so.
    """

    missing = [index for index, gradient in enumerate(grads) if gradient is None]
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(parameters)} trainable parameters received no "
            "gradient from the single-position loss, so the reported norm would "
            "not cover every parameter. The model's graph is not what this "
            "measurement assumes."
        )


def masked_evaluation_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    vocab_size: int,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Run one forward pass and return support-masked logits, graph intact.

    The restriction and masking sequence is the one
    :func:`~llm_behavior_lab.evaluation.init_distribution.measure_initialization`
    applies, in the same order, so the gradient observable and the guessing
    observables read literally the same masked logits rather than two
    independently-derived versions of them.

    The clone matters twice over: the mask is applied in place, and the model may
    hand back a view of its own buffer. Cloning is differentiable, so the graph
    is unaffected.

    Args:
        model: Model whose forward accepts ``input_ids`` and returns ``.logits``.
        input_ids: Token windows shaped ``[batch, block]``.
        vocab_size: Tokenizer vocabulary; logits wider than this are truncated.
        mask: Boolean ``[vocab_size]`` predictive-support mask.

    Returns:
        Masked logits shaped ``[batch, block, vocab_size]``.
    """

    output = model(input_ids=input_ids)
    logits = restrict_to_support(output.logits, vocab_size=vocab_size)
    return apply_support_mask(logits.clone(), mask)


def single_position_losses(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    vocab_size: int,
    eligible_token_ids: Sequence[int] | None = None,
) -> torch.Tensor:
    """Return ``ell_d`` for every position of every supplied window.

    This is the quantity the gradient is taken of: **one position's** next-token
    cross-entropy, not the window mean the model's own ``forward`` returns when
    given ``targets``. The two are related by ``mean(single_position_losses(...))
    == model(input_ids, targets=...).loss`` over an unrestricted support, which
    is exactly what the accompanying test asserts.

    Args:
        model: Model whose forward accepts ``input_ids`` and returns ``.logits``.
        input_ids: Token windows shaped ``[batch, block]``.
        target_ids: Next-token targets with the same shape.
        vocab_size: Tokenizer vocabulary.
        eligible_token_ids: Predictive support. ``None`` means every token.

    Returns:
        Losses shaped like ``target_ids``.
    """

    if input_ids.shape != target_ids.shape:
        raise ValueError(
            f"input_ids {tuple(input_ids.shape)} and target_ids "
            f"{tuple(target_ids.shape)} must have the same shape."
        )

    mask = eligible_support_mask(vocab_size, eligible_token_ids, device=input_ids.device)
    _check_targets_eligible(target_ids, mask)
    logits = masked_evaluation_logits(model, input_ids, vocab_size=vocab_size, mask=mask)
    losses = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        reduction="none",
    )
    return losses.reshape(target_ids.shape)


def _check_targets_eligible(target_ids: torch.Tensor, mask: torch.Tensor) -> None:
    """Fail loudly if a target token is outside the predictive support.

    An ineligible target has exactly zero probability under the masked softmax,
    so ``ell_d`` would be ``+inf`` and ``g_d`` meaningless. The experiment runner
    already refuses a corpus containing structural tokens; this is the backstop
    for any other caller.
    """

    flat = target_ids.reshape(-1)
    if bool((flat < 0).any()) or bool((flat >= mask.shape[0]).any()):
        raise ValueError("target_ids contain values outside the vocabulary.")
    ineligible = flat[~mask.to(flat.device)[flat]]
    if ineligible.numel():
        offenders = sorted(set(int(value) for value in ineligible.tolist()))[:5]
        raise ValueError(
            f"Target token IDs {offenders} are outside the predictive support, so "
            "-log P(y_d | context_d) is infinite. The corpus must not contain "
            "structural tokens."
        )


def compute_position_gradient_norms(
    model: torch.nn.Module,
    positions: EvaluationPositions,
    *,
    vocab_size: int,
    eligible_token_ids: Sequence[int] | None = None,
    window_indices: Sequence[int] | None = None,
    num_windows: int | None = None,
    temperatures: Sequence[float] = GRADIENT_TEMPERATURES,
    vector_split: bool = False,
    gradient_sketch: bool = False,
    exact_gradient_positions: Sequence[int] | None = None,
    sketch_dimension: int = DEFAULT_SKETCH_DIMENSION,
    sketch_seed: int = 20240917,
    progress: Callable[[int, int], None] | None = None,
    sketch_maps: int = 1,
    row_sink: Any = None,
) -> PositionGradientResult:
    """Measure ``g_d`` exactly, one evaluation position at a time.

    One forward pass per window feeds ``block_size`` backward passes taken from
    the retained graph. Peak memory is therefore one window's activations plus
    one full gradient set, independent of how many positions are evaluated.

    The model is never modified. Gradients come from :func:`torch.autograd.grad`,
    so ``parameter.grad`` is not touched, and the entry ``training`` mode is
    restored on every exit path.

    Args:
        model: Model whose forward accepts ``input_ids`` and returns ``.logits``.
        positions: The experiment's shared evaluation positions.
        vocab_size: Tokenizer vocabulary.
        eligible_token_ids: Predictive support. ``None`` means every token.
        window_indices: Explicit window subset. Defaults to every window.
        num_windows: Convenience alternative to ``window_indices``: keep this
            many evenly spaced windows. Ignored when ``window_indices`` is given.
        vector_split: Also accumulate ``g_correct`` and ``g_wrong``, the summed
            parameter gradients of correctly and incorrectly assigned positions
            at the canonical temperature. Off by default, so every existing
            caller keeps its exact cost and behaviour. Adds no backward pass --
            it reuses each gradient set already computed for the norm -- but
            holds two float64 parameter-shaped buffers, about 131 MiB at 8.6M
            parameters.
        exact_gradient_positions: Flat position indices whose **complete**
            gradient is copied to CPU for the CountSketch fidelity check. Off
            unless given. The copy is taken from the same gradient set the norm
            and the sketch already used, so no gradient is recomputed and no
            second backward pass is introduced.
        gradient_sketch: Also record a deterministic count sketch of every
            position's canonical gradient, so directional structure can be
            measured afterwards. Off by default. Adds no backward pass.
        sketch_dimension: Width ``K`` of that sketch.
        sketch_seed: Seed of the projection. Fixed across runs so two
            experiments produce comparable sketches; drawn from its own
            generator, never the global RNG. This is the **base** seed: map 0
            uses it unchanged, so a single-map run is unaffected by the replica
            schedule existing.
        progress: Optional ``callback(windows_done, windows_total)``.
        row_sink: Optional destination for the projected rows, implementing
            ``write_row(temperature_index, position, [M, K] float32)`` and
            ``seal()``. Given one, this function streams each position's rows to
            it and does **not** allocate the ``[N_T, D, M, K]`` array -- which is
            3.5 GiB at experiment scale and is the single largest allocation the
            measurement makes. Omitted, an in-memory sink is used and the arrays
            come back on the result exactly as they always have.

            The sink is deliberately the only thing this function knows about
            storage. It learns nothing about run directories, manifests,
            artifact formats or storage modes, so the projection path emits the
            same quantized rows in the same order whatever the destination is.
        sketch_maps: Number of independent production CountSketch maps ``M``.
            One by default, which is what every existing caller gets and what
            the runner still asks for. Above one, each position's gradient is
            projected through every map -- from the **same** ``autograd.grad``
            result, so the backward-pass count does not change -- and the sketch
            arrays gain a replica axis. Requires ``gradient_sketch``.

    Returns:
        A :class:`PositionGradientResult` with one entry per evaluated position.

        The sketch arrays are rank-conditional, and deliberately so. At ``M = 1``
        they keep their historical shapes exactly -- ``[N_T, D_g, K]`` and
        ``[D_g, K]`` -- with no length-one replica axis to squeeze, so nothing
        downstream sees a change. At ``M > 1`` they become ``[N_T, D_g, M, K]``
        and ``[D_g, M, K]``. **Read the map count from
        ``sketch_protocol["map_count"]``, never from the array's rank or width.**

    Raises:
        ValueError: If ``sketch_maps`` is not a positive integer, or exceeds one
            while ``gradient_sketch`` is off.
    """

    if window_indices is None:
        window_indices = evenly_spaced_indices(positions.num_windows, num_windows)
    else:
        window_indices = tuple(int(index) for index in window_indices)
        if not window_indices:
            raise ValueError("window_indices must not be empty.")
        if len(set(window_indices)) != len(window_indices):
            raise ValueError("window_indices must be distinct.")
        if min(window_indices) < 0 or max(window_indices) >= positions.num_windows:
            raise ValueError(
                f"window_indices must lie in [0, {positions.num_windows})."
            )

    grid = tuple(dict.fromkeys(float(value) for value in temperatures))
    if not grid:
        raise ValueError("temperatures must not be empty.")
    if any(value <= 0.0 for value in grid):
        raise ValueError(
            "Every gradient temperature must be positive; softmax(z/T) is "
            "undefined at T = 0."
        )
    if CANONICAL_GRADIENT_TEMPERATURE not in grid:
        raise ValueError(
            f"The canonical baseline T = {CANONICAL_GRADIENT_TEMPERATURE} must be "
            "present in the grid; it is the established observable every other "
            "temperature is read against."
        )

    device = positions.input_ids.device
    mask = eligible_support_mask(vocab_size, eligible_token_ids, device=device)
    _check_targets_eligible(positions.target_ids[list(window_indices)], mask)

    parameters = _trainable_parameters(model)
    block_size = positions.block_size
    total_positions = len(window_indices) * block_size

    position_indices = torch.empty(total_positions, dtype=torch.long)
    target_ids = torch.empty(total_positions, dtype=torch.long)
    greedy_ids = torch.empty(total_positions, dtype=torch.long)
    # [N_T, D_g] scalars only. A [D_g, N_T, parameters] tensor is never formed:
    # at 32768 positions, 7 temperatures and 8.6M parameters that would be about
    # 7.9 TB in float32. One gradient set exists at a time and is reduced to a scalar.
    gradient_norms = torch.empty((len(grid), total_positions), dtype=torch.float64)
    losses = torch.empty((len(grid), total_positions), dtype=torch.float64)
    split = _VectorSplitAccumulator(parameters) if vector_split else None
    wanted = (
        {int(index) for index in exact_gradient_positions}
        if exact_gradient_positions is not None
        else set()
    )
    captured: dict[int, torch.Tensor] = {}
    captured_sketches: dict[int, torch.Tensor] = {}
    map_count = _validated_map_count(sketch_maps, name="sketch_maps")
    if map_count != 1 and not gradient_sketch:
        # Refused rather than ignored. Silently measuring one map after being
        # asked for four would make the resulting record claim a replica budget
        # it never had, and the error would only surface as a missing axis much
        # further downstream.
        raise ValueError(
            f"sketch_maps={map_count} was requested but gradient_sketch is off, "
            "so no map would be built at all. Enable gradient_sketch or leave "
            "sketch_maps at 1."
        )
    sketcher = (
        _GradientSketcher(
            parameters,
            dimension=sketch_dimension,
            seed=sketch_seed,
            map_count=map_count,
        )
        if gradient_sketch
        else None
    )
    # One tensor for every temperature. The canonical row is taken from it at
    # the end rather than accumulated separately, so no second T = 1 gradient
    # field can drift away from this one.
    #
    # The replica axis is present only when there are replicas: at M = 1 the
    # shape is exactly what it has always been, rather than a [.., 1, K] block
    # that would have to be squeezed on the way out. A squeeze is one more place
    # for a stray axis to reach a record, and the M = 1 arrays are the ones every
    # existing consumer and the frozen fixture depend on.
    #
    # The rows go to a sink rather than into an array owned here. At experiment
    # scale that array is 3.5 GiB, and holding it was the reason a production arm
    # could not be scaled; the sink lets the same rows stream to bounded disk
    # instead, without this loop knowing that is where they went.
    sink = row_sink
    if gradient_sketch and sink is None:
        from llm_behavior_lab.evaluation.sketch_store import (
            InMemoryRowSink,
            SketchStoreLayout,
        )

        sink = InMemoryRowSink(
            SketchStoreLayout(
                num_temperatures=len(grid),
                num_positions=total_positions,
                num_maps=map_count,
                num_buckets=sketch_dimension,
            )
        )
    if sink is not None and not gradient_sketch:
        raise ValueError(
            "A row_sink was given but gradient_sketch is off, so no row would "
            "ever be projected into it."
        )
    canonical_temperature_index = grid.index(CANONICAL_GRADIENT_TEMPERATURE)

    was_training = model.training
    started = time.perf_counter()
    try:
        model.eval()
        cursor = 0
        for done, window in enumerate(window_indices, start=1):
            window_inputs = positions.input_ids[window : window + 1]
            window_targets = positions.target_ids[window : window + 1]
            logits = masked_evaluation_logits(
                model, window_inputs, vocab_size=vocab_size, mask=mask
            )[0]
            # Read from the same masked logits as the loss, in the same forward
            # pass, so the guess fractions are position-aligned with the norms
            # by construction rather than by a second, separately-batched pass.
            window_greedy = greedy_guess_ids(logits.detach())

            last_offset = block_size - 1
            last_temperature = len(grid) - 1
            for offset in range(block_size):
                position_indices[cursor] = window * block_size + offset
                target_ids[cursor] = window_targets[0, offset]
                greedy_ids[cursor] = window_greedy[offset]

                for index, temperature in enumerate(grid):
                    # Temperature scales the logits *inside* the loss, so the
                    # gradient really is the gradient of a different objective --
                    # not a rescaled version of one. Dividing by exactly 1.0 is
                    # the identity in IEEE-754, so the canonical row is bitwise
                    # the established T = 1 observable rather than a near copy.
                    loss = F.cross_entropy(
                        logits[offset : offset + 1] / temperature,
                        window_targets[0, offset : offset + 1],
                    )
                    # Not .backward(): autograd.grad returns the gradients and
                    # leaves every parameter.grad exactly as the caller left it.
                    # The one forward graph for this window is retained until the
                    # final temperature of the final position uses it.
                    grads = torch.autograd.grad(
                        loss,
                        parameters,
                        retain_graph=not (
                            offset == last_offset and index == last_temperature
                        ),
                        allow_unused=True,
                    )
                    _require_every_parameter_reached(grads, parameters)
                    # Accumulated in float64 on whichever device the gradients
                    # live on; an 8.6M-parameter sum of squares loses meaningful
                    # precision in float32.
                    squared = None
                    for gradient in grads:
                        contribution = gradient.detach().double().pow(2).sum()
                        squared = contribution if squared is None else squared + contribution

                    gradient_norms[index, cursor] = squared.sqrt().cpu()
                    losses[index, cursor] = loss.detach().double().cpu()
                    if sketcher is not None:
                        # Same gradient set the norm came from, at this very
                        # temperature; no second backward pass and nothing
                        # full-sized is retained.
                        #
                        # The single-map branch calls the historical `project`
                        # directly rather than `project_maps(grads)[0]`. The two
                        # give the same numbers, but routing M = 1 through the
                        # replica path would put a stack-and-index between the
                        # established observable and its own code, for nothing.
                        #
                        # `.cpu()` returns float64; the float32 conversion below
                        # is the same narrowing the preallocated buffer used to
                        # perform on assignment, in the same place in the order
                        # of operations. Nothing about the quantization moved.
                        projected = (
                            sketcher.project(grads).cpu().reshape(1, -1)
                            if map_count == 1
                            else sketcher.project_maps(grads).cpu()
                        )
                        rows_f32 = projected.to(torch.float32)
                        sink.write_row(index, cursor, rows_f32.numpy())
                        if (
                            wanted
                            and index == canonical_temperature_index
                            and int(position_indices[cursor]) in wanted
                        ):
                            # The sketch the production maps actually produced
                            # for this position, kept alongside its exact
                            # gradient. Rebuilding the maps afterwards to
                            # re-project would measure a reconstruction of the
                            # instrument rather than the instrument.
                            captured_sketches[int(position_indices[cursor])] = (
                                rows_f32.clone()
                            )
                    if (
                        wanted
                        and index == canonical_temperature_index
                        and int(position_indices[cursor]) in wanted
                    ):
                        # The same live gradient set the norm and the sketch came
                        # from, flattened in parameter order and moved to CPU one
                        # position at a time. Nothing accumulates on the GPU.
                        captured[int(position_indices[cursor])] = torch.cat(
                            [
                                gradient.detach().reshape(-1).float().cpu()
                                for gradient in grads
                            ]
                        )
                    if split is not None and index == canonical_temperature_index:
                        # The same gradient set the norm was taken from, before
                        # it is released. No second backward pass.
                        split.add(
                            grads,
                            is_correct=bool(
                                int(greedy_ids[cursor]) == int(target_ids[cursor])
                            ),
                        )
                    del grads, squared, loss
                cursor += 1

            del logits, window_greedy
            if progress is not None:
                progress(done, len(window_indices))
    finally:
        model.train(was_training)
    seconds = time.perf_counter() - started

    canonical = grid.index(CANONICAL_GRADIENT_TEMPERATURE)

    # Seal before anything reads the rows. For a temporary store this is what
    # validates sizes, row counts and digests; for the in-memory sink it checks
    # that every position was written. Either way the arrays below are only
    # reachable once the sink says the collection is complete.
    temperature_sketches = None
    if sink is not None:
        sink.seal()
        # Only an in-memory sink can hand back arrays. A store deliberately
        # cannot: its whole purpose is that the rows never enter the record.
        temperature_sketches = getattr(sink, "temperature_sketches", None)
        if temperature_sketches is not None:
            temperature_sketches = torch.from_numpy(temperature_sketches)

    return PositionGradientResult(
        position_indices=position_indices,
        target_ids=target_ids,
        greedy_ids=greedy_ids,
        temperature_gradient_norms=gradient_norms,
        temperature_losses=losses,
        temperatures=grid,
        gradient_norms=gradient_norms[canonical],
        losses=losses[canonical],
        vector_split=None if split is None else split.summary(),
        temperature_gradient_sketches=temperature_sketches,
        gradient_sketches=(
            None if temperature_sketches is None else temperature_sketches[canonical]
        ),
        exact_gradients=(
            None
            if not captured
            else torch.stack([captured[key] for key in sorted(captured)])
        ),
        sketch_tensor_sizes=(
            None if sketcher is None else list(sketcher.tensor_sizes)
        ),
        exact_positions=(
            None
            if not captured
            else torch.tensor(sorted(captured), dtype=torch.long)
        ),
        exact_sketches=(
            None
            if not captured_sketches
            else torch.stack(
                [captured_sketches[key] for key in sorted(captured_sketches)]
            )
        ),
        sketch_map=None if sketcher is None else sketcher.numpy_map(),
        sketch_protocol=(
            None
            if sketcher is None
            else {
                # Every key below this line predates replicas and keeps its
                # exact meaning. `seed` in particular is the historical
                # production base seed and is not renamed: artifacts and readers
                # already refer to it by that name.
                "dimension": sketcher.dimension,
                "seed": sketcher.seed,
                "temperature": CANONICAL_GRADIENT_TEMPERATURE,
                "projection": "count_sketch_signed_feature_hashing",
                "preserves": "inner_products_in_expectation",
                "parameter_count": sum(p.numel() for p in parameters),
                # Additive, and written at M = 1 too. Single-map operation was
                # previously knowable only from the source that produced the
                # artifact; recording it closes that provenance gap for every
                # newly written record, and gives downstream code an explicit
                # count to branch on instead of an array's rank.
                "map_count": sketcher.map_count,
                "base_seed": sketcher.seed,
                # Read off the bank the sketcher actually built, not recomputed
                # from the formula here -- a duplicated derivation is exactly how
                # recorded seeds and real seeds drift apart.
                "map_seeds": [int(value) for value in sketcher._map_seeds],
                "seed_derivation": _SKETCH_SEED_DERIVATION,
                "seed_derivation_version": _SKETCH_SEED_DERIVATION_VERSION,
                # Identifies the scalar domain the map was drawn for. A stored
                # map is only meaningful against the layout it was built over,
                # and `parameter_count` alone cannot distinguish two models that
                # happen to have the same total while ordering or naming their
                # tensors differently.
                "parameter_layout_sha256": _parameter_layout_sha256(
                    model, parameters
                ),
                "parameter_layout_schema": _PARAMETER_LAYOUT_SCHEMA,
                "parameter_ordering": _PARAMETER_ORDERING,
                # Tensors, not scalars: `parameter_count` above stays the total
                # element count and is not redefined.
                "parameter_tensor_count": len(parameters),
                # -- protocol schema v2 -------------------------------------
                # How the arrays beside this protocol must be *read*. Exact
                # values rather than free text: a reader finding something
                # unexpected is looking at a layout it does not understand and
                # must say so rather than guess.
                "schema_version": _SKETCH_PROTOCOL_SCHEMA_VERSION,
                "canonical_storage": "stored" if map_count == 1 else "derived",
                "canonical_relationship": "slice_of_temperature_array",
                "temperature_sketch_axes": (
                    ["temperature", "position", "bucket"]
                    if map_count == 1
                    else ["temperature", "position", "map", "bucket"]
                ),
                "estimator": "mean_of_per_map_inner_products_over_exact_norms",
                "accumulation_dtype": "float64",
                "storage_dtype": "float32",
                # Every map projects the same gradient from one backward pass.
                # Recorded because it is the governing claim of the replica
                # design: a record saying otherwise describes a different
                # measurement, in which projection and recomputation variance
                # are confounded.
                "recomputed_per_map": False,
            }
        ),
        parameter_count=sum(parameter.numel() for parameter in parameters),
        num_parameter_tensors=len(parameters),
        num_windows=len(window_indices),
        block_size=block_size,
        seconds=seconds,
    )

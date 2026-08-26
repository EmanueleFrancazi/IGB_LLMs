"""Scale the random part of a model's initialization by a global factor.

The intervention is one multiplier ``alpha`` applied to every trainable tensor
that PyTorch draws from a zero-centred random distribution:

.. code-block:: text

    W^(alpha) = alpha * W^(1)      for the audited stochastic tensors
    W^(alpha) = W^(1)              for everything else

Both models come from the **same draw**. The baseline is built once from its
seed and the multiplier is applied to that realization, so the three scale
conditions share signs, directions and relative structure exactly and differ
only in magnitude. Reseeding per scale would give three unrelated draws and
answer a different question.

**There is no single architecture-wide ``sigma_w`` here, and ``alpha`` is not a
rescaling of one.** This model inherits PyTorch's per-class defaults, which
differ in both scale and distribution family: the token embedding is
``normal_(0, 1)``, while every ``nn.Linear`` is ``kaiming_uniform_(a=sqrt(5))``,
whose standard deviation depends on fan-in -- about 0.051 at fan-in 128 and
0.029 at fan-in 384. ``alpha`` multiplies all of them by a common factor, so
every group's standard deviation moves as ``1 : 1/2 : 1/4`` while the ratios
*between* groups are preserved. When relating this to work that intervenes on a
single ``sigma_w``, ``alpha`` is the analogue of that intervention, not the same
thing.

**Why this need not be a pure logit rescaling.** ``logits = output(norm(h))``,
and RMSNorm divides out the scale of ``h``, so ``alpha`` reaches the logits only
through ``output.weight`` -- exactly a common factor -- *provided the direction*
of ``h`` is unchanged. It generally is not: ``h`` mixes the embedding, which
scales as ``alpha``, with the attention and feed-forward branches, which scale
as ``alpha^2`` (two projections in series), and attention scores scale as
``alpha^2`` as well, so the softmax pattern itself sharpens or flattens. The
residual-to-branch ratio therefore changes with ``alpha``. This is mechanistic
motivation for why greedy identities *may* move; it is not a prediction, and
nothing here or in the tests assumes an outcome.

The architecture contains **no bias parameters at all** -- every ``nn.Linear`` is
built ``bias=False`` -- so the "biases fixed at zero" condition is identical to
every historical experiment rather than a new constraint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from torch import nn

__all__ = [
    "DETERMINISTIC_PARAMETER_SUFFIXES",
    "ParameterGroupReport",
    "classify_parameters",
    "deterministic_parameter_suffixes",
    "initialization_scale_report",
    "scale_initialization",
]

#: Trainable tensors initialized deterministically rather than from a
#: zero-centred random distribution. The RMSNorm gains are set to exactly one;
#: multiplying them by ``alpha`` would change the normalization itself, which is
#: a different intervention from scaling the random draw.
#:
#: These names belong to the LLaMA-family architecture, and they are the default
#: for any model that does not say otherwise. A family whose deterministic
#: tensors have different names declares its own set -- see
#: :func:`deterministic_parameter_suffixes`.
DETERMINISTIC_PARAMETER_SUFFIXES = (
    "norm.weight",
    "attention_norm.weight",
    "ffn_norm.weight",
)


def deterministic_parameter_suffixes(model: nn.Module) -> tuple[str, ...]:
    """Return the deterministic-parameter suffixes that apply to ``model``.

    A model class may declare a class attribute of the same name as the module
    constant to override it:

    .. code-block:: python

        class SomeFamilyForCausalLM(BaseLanguageModel):
            DETERMINISTIC_PARAMETER_SUFFIXES = ("ln_f.weight", "ln_f.bias")

    A model that declares nothing gets :data:`DETERMINISTIC_PARAMETER_SUFFIXES`,
    which is the historical behaviour on exactly the historical code path. This
    is why the LLaMA family declares nothing: falling through is identical by
    construction rather than identical by careful transcription.

    **Why a per-family declaration and not one global rule.** The deterministic
    set is a property of how a family initializes itself, and the families
    disagree. LLaMA-style RMSNorm gains are ``ones_`` and carry no bias at all,
    so three suffixes cover it. A GPT-2-style family additionally has LayerNorm
    biases and zero-initialized ``nn.Linear`` biases, and it names its norms
    ``ln_1`` / ``ln_2`` / ``ln_f``. Excluding *every* ``.bias`` globally would be
    wrong: a family that initializes a bias from a zero-centred random draw must
    have it scaled, and a blanket rule would silently stop scaling it.

    **Classification is by name, never by value.** Reading a tensor to see
    whether it happens to hold zeros or ones would misclassify a random draw that
    landed near a constant, would depend on the seed, and would make the audit
    describe the realization instead of the intent.

    An explicitly empty declaration is honoured: it means the family has no
    deterministic trainable tensors, which is different from declaring nothing.

    Raises:
        TypeError: If the declaration is a bare string, or holds anything other
            than non-empty strings. A bare string is the dangerous case --
            iterating it would test single characters and quietly exclude far
            more than intended.
    """

    declared = getattr(model, "DETERMINISTIC_PARAMETER_SUFFIXES", None)
    if declared is None:
        return DETERMINISTIC_PARAMETER_SUFFIXES

    if isinstance(declared, str):
        raise TypeError(
            "DETERMINISTIC_PARAMETER_SUFFIXES must be a sequence of suffixes, not a "
            f"single string; got {declared!r}. Iterating a string would match "
            "individual characters and exclude tensors that should be scaled."
        )

    suffixes = tuple(declared)
    if not all(isinstance(suffix, str) and suffix for suffix in suffixes):
        raise TypeError(
            "Every entry of DETERMINISTIC_PARAMETER_SUFFIXES must be a non-empty "
            f"string; got {suffixes!r}."
        )
    return suffixes


@dataclass(frozen=True)
class ParameterGroupReport:
    """Audit of one trainable tensor before and after the intervention."""

    name: str
    shape: tuple[int, ...]
    numel: int
    is_random: bool
    baseline_std: float
    effective_std: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable description."""

        return {
            "name": self.name,
            "shape": list(self.shape),
            "numel": self.numel,
            "scaled": self.is_random,
            "baseline_std": self.baseline_std,
            "effective_std": self.effective_std,
            "effective_variance": self.effective_std**2,
        }


def _is_deterministic(name: str, suffixes: Sequence[str]) -> bool:
    return any(name.endswith(suffix) for suffix in suffixes)


def classify_parameters(model: nn.Module) -> dict[str, bool]:
    """Map each trainable parameter name to whether the scale applies to it.

    Classification is by name against the deterministic set the model's family
    declares -- :func:`deterministic_parameter_suffixes` -- so a tensor added
    later is scaled by default and has to be excluded deliberately. That is the
    safer direction to fail: a new random tensor silently left unscaled would
    break the pairing without any symptom.

    Insertion order follows ``model.named_parameters()`` and is unaffected by
    which set applies.
    """

    suffixes = deterministic_parameter_suffixes(model)
    classified: dict[str, bool] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        classified[name] = not _is_deterministic(name, suffixes)
    return classified


def _unique_storages(model: nn.Module) -> dict[int, str]:
    """Map storage identity to the first parameter name using it.

    Tied weights share one tensor under two names. Scaling by name alone would
    then multiply the same storage twice and silently produce ``alpha^2``. This
    architecture ties nothing, but the guard is what makes that a checked fact
    rather than an assumption.
    """

    seen: dict[int, str] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        key = parameter.data_ptr()
        seen.setdefault(key, name)
    return seen


def scale_initialization(model: nn.Module, alpha: float) -> dict[str, Any]:
    """Multiply the audited random tensors of ``model`` in place by ``alpha``.

    ``alpha = 1`` is a **literal no-op**: the function touches nothing at all,
    rather than multiplying by 1.0. The baseline condition is then the ordinary
    model, unmodified, and cannot differ from it by any means.

    Apply this once to a freshly built model. Applying it twice would compound
    the factor; each scale condition must be derived independently from its own
    pristine baseline draw.

    Args:
        model: A freshly initialized model.
        alpha: Positive multiplier for the random tensors.

    Returns:
        A JSON-serializable description of what was and was not scaled.

    Raises:
        ValueError: If ``alpha`` is not positive.
    """

    if not alpha > 0.0:
        raise ValueError(f"alpha must be positive; got {alpha}.")

    classified = classify_parameters(model)
    first_use = _unique_storages(model)
    scaled: list[str] = []
    unscaled: list[str] = []

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if not classified[name]:
                unscaled.append(name)
                continue
            if first_use[parameter.data_ptr()] != name:
                # An alias of a tensor already handled: scaling it again would
                # square the factor.
                continue
            scaled.append(name)
            if alpha != 1.0:
                parameter.mul_(alpha)

    return {
        "alpha": float(alpha),
        "is_no_op": alpha == 1.0,
        "scaled_parameters": scaled,
        "unscaled_parameters": unscaled,
        "num_scaled_tensors": len(scaled),
        "num_scaled_elements": int(
            sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if name in set(scaled)
            )
        ),
        "note": (
            "alpha multiplies every audited zero-centred random tensor. There is "
            "no single architecture-wide sigma_w: the embedding is normal_(0,1) "
            "and every linear is kaiming_uniform_(a=sqrt(5)) with a fan-in "
            "dependent scale, so alpha is the analogue of a sigma_w intervention "
            "rather than a rescaling of one value."
        ),
    }


def initialization_scale_report(
    model: nn.Module,
    alpha: float,
    *,
    baseline: Iterable[tuple[str, torch.Tensor]] | None = None,
) -> dict[str, Any]:
    """Describe the per-group standard deviations after the intervention.

    Reported per parameter group rather than as one number, because the model
    has no architecture-wide standard deviation to report.

    Args:
        model: The scaled model.
        alpha: The multiplier that was applied.
        baseline: Optional ``(name, tensor)`` pairs from the unscaled model. When
            given, each group's baseline standard deviation is measured from it
            rather than inferred by dividing, which keeps the check independent
            of the operation being checked.

    Returns:
        A JSON-serializable per-group report.
    """

    baseline_std = (
        {name: float(tensor.detach().float().std().item()) for name, tensor in baseline}
        if baseline is not None
        else {}
    )
    classified = classify_parameters(model)

    groups: list[ParameterGroupReport] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        effective = float(parameter.detach().float().std().item())
        if name in baseline_std:
            original = baseline_std[name]
        elif classified[name]:
            original = effective / alpha if alpha else float("nan")
        else:
            original = effective
        groups.append(
            ParameterGroupReport(
                name=name,
                shape=tuple(parameter.shape),
                numel=int(parameter.numel()),
                is_random=classified[name],
                baseline_std=original,
                effective_std=effective,
            )
        )

    return {
        "alpha": float(alpha),
        "variance_factor": float(alpha) ** 2,
        "has_single_sigma_w": False,
        "groups": [group.as_dict() for group in groups],
        "num_scaled_elements": sum(g.numel for g in groups if g.is_random),
        "num_unscaled_elements": sum(g.numel for g in groups if not g.is_random),
    }

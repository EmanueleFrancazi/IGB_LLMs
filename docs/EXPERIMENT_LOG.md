# Experiment log — initialization token-guess distributions

The scientific record for the pre-Phase-7 initialization-distribution
experiment: what it asks, how every quantity is defined, and what has actually
been observed so far.

Written for a reader returning after months. [`README.md`](../README.md) is the
orientation document; this is the authoritative explanation.

---

## 1. The question

> At random initialization, how do the distributions of the model's **token
> guesses** compare with the empirical token distribution of the corpus — in
> overall concentration and token by token — and how stable is that comparison
> across independent model initializations?

The model is **untrained**. Nothing here measures language quality. Results
describe the interaction between an untrained architecture, its initialization
scheme, and the corpus it is compared against.

### Interpretation boundary with a pretrained tokenizer

The subword runs use the pretrained `mistralai/Mistral-7B-v0.1` **tokenizer**.
No pretrained model weights are ever loaded; the model is the project's own
randomly initialized tiny LLaMA.

The tokenizer nonetheless carries linguistic structure learned from its own
training corpus: the segmentation, the frequency profile of the pieces, and
which strings are single tokens at all. So the subword experiment measures

> the distributional behavior of a randomly initialized tiny LLaMA **over a
> realistic pretrained subword vocabulary**,

not a completely unlearned text-processing system. Structure visible in the
*corpus* token distribution belongs to the tokenizer and the text. Only the
*guess* distribution belongs to the random model. Do not attribute the former to
the latter.

---

## 2. Nomenclature

| Symbol | Meaning |
|---|---|
| `i` | canonical token identity (the token ID the tokenizer assigns) |
| `s` | one independent model initialization |
| `r` | rank after sorting, where a quantity is ranked |
| `N` | evaluation positions — selections made by greedy, and by each nucleus replicate |
| `I` | number of independent model initializations |
| `R` | nucleus sampling replicates per initialization |

### Distributions

| Symbol | Definition |
|---|---|
| `p_i` | empirical token fraction over the **whole selected corpus split** |
| `p_selected_i` | empirical **next-token** fraction over the fixed selected evaluation positions |
| `q_{s,i}` | **selected-guess** frequency for token `i` under initialization `s` |

`p` is the primary reference the guesses are compared against.
`p_selected` is a **sampling-adequacy diagnostic**, not a model reference — it
answers whether the analyzed positions stand in for the split.

When the analysis split is `train`, `p` covers the entire training split;
validation tokens are excluded.

### Not to be confused

`q_{s,i}` is a **selected-guess frequency**: how often a policy actually chose
token `i`. It is not the mean predicted probability, which is the average mass
the model places on `i` across positions. Both are persisted, under separate
names, and they diverge sharply at initialization.

### Vocabulary sizes

Four different questions, four different numbers. Only the first three are
counts.

| Quantity | Question |
|---|---|
| `V` full | how many tokens the tokenizer defines |
| `V` eligible | how many a model may be scored on, after excluding structural IDs |
| `V` corpus-observed | how many actually occur in the analysis split |
| `N_eff = exp(H)` | how broadly the mass is spread |

Structural tokens (BOS, EOS, UNK) are **excluded** from the predictive support,
never renumbered. Canonical IDs mean the same thing here as upstream.

---

## 3. Protocol

Everything except the model-initialization seed is held fixed: corpus,
tokenizer, vocabulary, analysis split, and the evaluation positions themselves.
Positions are chosen deterministically — evenly spaced starts, no RNG consumed —
*before* any model exists, so no difference between initializations can come
from looking at different text.

```
fixed corpus + fixed evaluation positions
            ↓
   multiple random model initializations
            ↓
        same logits (one forward pass per initialization)
       ↙                    ↘
   greedy              nucleus sampling
                            ↓
              R controlled replicates (R = 1 now)
       ↘                    ↙
     complete per-token result records
            ↓
   cross-initialization aggregation
            ↓
        saved figures
```

### Guessing policies

Both read the **same** logits, so they describe one model state rather than two
independent draws.

| Policy | Rule | Randomness |
|---|---|---|
| greedy | `argmax(logits)` | none beyond the initialization |
| nucleus | temperature, then top-p truncation (reference LLaMA rule) | its own seed |

The nucleus rule drops a token when the cumulative mass *strictly before* it
exceeds `top_p`, so the most probable token always survives.

### Three separated sources of randomness

1. **Position sampling** — removed entirely; positions are deterministic.
2. **Model initialization** — the quantity under study, and the independent unit
   for every reported standard error.
3. **Token sampling** — affects only the nucleus policy and is driven by its own
   seed. With `R > 1` it is averaged over replicates *within* an initialization
   before initializations are compared. The current protocol uses `R = 1`; see
   §5 for what that does and does not allow.

Sampling draws are pre-drawn and indexed by position, so a streamed measurement
equals an all-at-once one exactly and the batch size cannot influence a result.

---

## 4. Measures

### Distance and shape

| Measure | Definition |
|---|---|
| Total variation | `TV(p, q) = ½ · Σ_i |p_i − q_i|`, bounded in `[0, 1]` |
| Jensen–Shannon | symmetric divergence in nats, bounded by `ln 2` |
| Entropy | `H(x) = −Σ_i x_i log x_i`, in nats |
| Effective support | `N_eff(x) = exp(H(x))` |
| Top1–top2 gap | `x_(1) − x_(2)` on the ranked distribution |

**Effective support** is the entropy-equivalent number of equally likely active
tokens. It is not a count of anything observed: a distribution touching 6000
tokens but concentrating on a handful has a small `N_eff`.

### Zero frequency, and the equal-draw correction

Zero-frequency count = eligible tokens never selected. Fraction = that count over
`V` eligible.

This statistic depends strongly on the number of draws, so a comparison between
policies is only meaningful at **equal draw counts**:

- greedy makes exactly one selection per position → `N` draws;
- each nucleus replicate also makes `N` draws;
- the `R` replicates **pooled** have `R·N` draws.

Pooling gives nucleus more chances to reach a rare token, so a pooled figure is
not comparable with greedy. The headline nucleus statistic is therefore
**per replicate**:

```
Z_{s,r} = #{ i ∈ V_eligible : q_{s,r,i} = 0 }        each over N draws
Z_s     = (1/R) · Σ_r Z_{s,r}                        averaged within initialization s
report    mean ± SEM of Z_s across initializations
```

The pooled value is retained separately as
`pooled_zero_frequency_count_mean`, with its own draw count. It answers a real
question — what the policy can reach given more attempts — but it is never shown
beside the greedy figure. Greedy has no pooled figure.

**Zero frequency and effective support are both kept.** One asks how much of the
vocabulary went untouched in a fixed number of draws; the other how broadly the
mass is spread. Neither replaces the other.

### Same-token mismatch

Token identity is preserved **before** ranking:

```
typical      d_{s,i} = |q_{s,i} − p_i|              then rank, then mean ± SEM over s
persistent            |mean_s(q_{s,i}) − p_i|       then rank
```

Ranking first and differencing afterwards would compare unrelated tokens: a
distribution carrying the corpus frequencies on the *wrong* tokens has an
identical ranked profile. Large typical with small persistent means
initializations disagree about which tokens they over-select; both large means
the bias is systematic.

### Variability

`SEM = s/√I` across independent initializations, with the sample standard
deviation (`ddof=1`). Reported for guess quantities only — the corpus
distribution is fixed, not a sample, and carries no band.

Within-initialization stochastic variability is reported separately from
between-initialization variability, so replicate noise is never presented as
initialization noise.

Note these describe different things and none of them substitutes for the
others: corpus-sample representativeness (figure 0), initialization variability
(the SEM bands), and sampling variability (the separate diagnostic). More
initializations cannot repair unrepresentative evaluation positions.

---

## 5. Null comparisons

Two reference families were added after the pilots. Both answer "compared with
what?", which the model measurements alone cannot.

### The uniform categorical output null

Let `K = V eligible` and let `D` be the evaluated position count. Under the null
every eligible token is chosen with probability `1/K`, and one realization is

```
(C_1, ..., C_K) ~ Multinomial(D; 1/K, ..., 1/K),    q_i^null = C_i / D
```

ranked descending. `M` independent realizations give a mean ranked profile and a
pointwise **Monte Carlo interval** (95% by default).

**The draw is joint, not marginal.** Each `C_i` is marginally
`Binomial(D, 1/K)`, but simulating `K` independent binomials would get the ranked
histogram wrong: the counts are negatively dependent and must sum to exactly `D`.
Independent binomials do neither, and produce a systematically over-spread
profile.

`M` is **not** `R`. It is `uniform_null.replicates`, defaults to 256, and is
persisted with the run. At `K = 31,997` it costs about 0.6 s and ~65 MB of
transient storage.

**Analytic checks.** For a uniform chooser, exactly:

```
E[Z]   = K (1 - 1/K)^D                                   zero-frequency count
E[Z]/K = (1 - 1/K)^D                                     zero-frequency fraction
E[N_j] = K C(D, j) (1/K)^j (1 - 1/K)^(D-j)               tokens seen exactly j times
```

These are unit checks on the simulation, never substitutes for it — none of them
produces a *ranked* profile. Measured agreement at `K = 31,997`, `D = 8192`:
Monte Carlo 24,769.7 against analytic 24,769.5.

**What it is not.** The null is not a model and has no initializations, so it has
no initialization SEM. Figure 1 draws it grey and dashed with a hatched band
labelled "not a SEM" precisely so the two kinds of uncertainty cannot be read as
the same thing.

### Input-structure conditions

One initialized model, three inputs, evaluated in the same run:

| Condition | Construction |
|---|---|
| real | the fixed corpus evaluation windows |
| shuffled | the **exact same multiset** of evaluated token IDs, globally permuted and reshaped |
| gaussian | synthetic vectors supplied at the embedding boundary, bypassing token lookup |

The shuffle preserves the position count, the token IDs, their counts, the
marginal token distribution, the tokenizer, the embedding table, the window
shapes, and the positional layout. It destroys local ordering and sequential
correlation, and nothing else. The permutation and the standardized Gaussian bank
are drawn from dedicated seeds and are **fixed across initializations**, so the
controls never vary with the weights.

Gaussian vectors are scale-matched per initialization: a standardized bank `Z` is
drawn once, then `G = mu + sigma * Z` using the realized entry-wise mean and
standard deviation of *that* initialization's **eligible** embedding rows.
Structural rows are excluded because real input never looks them up. The measured
`mu` and `sigma` are persisted per initialization.

**What each contrast can identify:**

| Contrast | Identifies |
|---|---|
| real vs shuffled | approximately **sequential ordering**, with everything else held fixed |
| shuffled vs gaussian | what **discrete token identity** adds once ordering is gone: repeated lookup vectors and unigram structure |
| real vs gaussian | the broadest contrast, structured tokens against unstructured continuous input |

**real vs gaussian must never be described on its own as a causal test of input
correlation.** It removes far more than temporal structure.

### R = 1 and common random numbers

**Selected explicitly by `configs/experiment/initialization_distribution.yaml`**,
which sets `num_replicates: 1` and `common_random_numbers: true`. The library
defaults are deliberately the *historical* ones (8 replicates, a
per-initialization sampling stream), so a configuration written before this
integration keeps its original meaning rather than silently acquiring a new
protocol.

The active protocol takes **one** nucleus draw per initialization and position.
Greedy and nucleus then produce exactly `D` assignments each, so both are
summarised at the same sample size and neither needs rescaling to be compared
with the other or with the null.

Consequences, all documented rather than papered over:

- **Within-initialization stochastic variance is not estimable.** It is reported
  as `not estimated (R=1)` with `None` values, never as a zero that would read as
  an observed absence of noise.
- The nucleus SEM across initializations describes the combined
  **"random initialization + one fixed stochastic sampling realization"**
  procedure, not initialization variability alone.
- `R = 1` SEMs are **not** comparable with the historical `R = 4` values in §7.4
  without noting the protocol change.

Sampling uniforms are drawn from a dedicated seed and indexed by **position
alone**, so the same position draws the same uniform in every input condition and
every initialization. That conditions the input comparison on one fixed
stochastic realization instead of letting sampling noise drift between
conditions, and it makes the result independent of `forward_batch_size`.

Multi-replicate support and the per-replicate zero-frequency correction (§4)
remain fully available for historical configurations, and remain what a config
omitting these fields resolves to.

### Paired input-condition measures

Ranked profiles discard token identity, so figure 4 alone cannot say whether two
conditions prefer *different* tokens. These paired measures answer that, computed
**within** an initialization and only then averaged:

- `tv_real_vs_shuffled`, `tv_real_vs_gaussian`, `tv_shuffled_vs_gaussian` —
  same-token total variation between conditions. These are **not** distances to
  the corpus and are named so they cannot be confused with `TV(corpus, guesses)`.
- paired differences in effective support, entropy, zero-frequency fraction, and
  the top1–top2 gap, e.g.
  `ΔN_eff^{real−shuffle} = N_eff(q_s^real) − N_eff(q_s^shuffle)`.

Pairing matters: the initialization contributes to both terms, so differencing
first removes it. Comparing two independent mean±SEM intervals instead would
discard that and inflate the apparent uncertainty.

### Figure 4

`figure4_input_structure_profiles.svg`, two panels (greedy, nucleus), three
ranked condition profiles each with initialization SEM, using the same ranking
convention as figure 1 — each condition ranked independently within an
initialization, then aggregated rank by rank. **Token identity is discarded
here too**; the paired measures above are what preserve it.

Figures 0–3 keep their roles unchanged and describe the **real** condition only.
The uniform null is added to figure 1 alone: it is exchangeable across token
identities, so it belongs with the ranked/concentration comparison and would be
meaningless on the identity-preserving figures 2 and 3.

---

## 5c. The temperature sweep

Optional additional analysis. The canonical policy remains the scalar
`sampling.temperature`, and figures 0-4 are unchanged; the sweep appears only in
figure 5.

### Definition

For a set of **strictly positive** temperatures, the same logits are sampled
once per temperature at the same fixed `top_p`. Greedy is the `T = 0` anchor and
is computed by `argmax`, never by a zero-temperature softmax, which is
undefined. The question is:

> As temperature increases at fixed `top_p = 0.9`, how does the selected-guess
> distribution move from the greedy concentration profile toward the finite-`D`
> uniform categorical null?

Nothing here presumes a sharp critical temperature. The transition is whatever
the measurements show, and monotonicity is an empirical question rather than a
property finite stochastic samples under top-p truncation are guaranteed to have.

The documented first sweep is `T = 0.12, 0.24, 0.36, 0.48, 0.60, 1.20`, with
`T = 0` as the greedy anchor.

### One forward pass, one sort, one draw

Temperature is a positive scale, so it cannot reorder logits: `x_i > x_j` implies
`x_i/T > x_j/T`. That gives three reuses, all of them load-bearing:

* **the forward pass** — the model runs once per initialization and input
  condition, exactly as before. Every temperature is derived from the batch of
  logits already in hand, before it is discarded. Streaming and the flat-memory
  guarantee are untouched;
* **the sort** — one `argsort` serves the whole sweep. Probabilities are computed
  in the original order and *gathered* by that permutation, which moves floats
  without arithmetic. Recomputing the softmax on reordered logits would not be
  safe: the denominator is a sum, and summation order can change the last ulp;
* **the uniforms** — common random numbers now extend to temperature. Position
  `d` draws the same uniform at every temperature, in every input condition, and
  in every initialization, so a difference between temperatures cannot be
  sampling noise.

### Exact equality at the canonical temperature

When the sweep contains the canonical temperature, its count vector is
**bit-for-bit identical** to the canonical nucleus counts. Both paths share one
implementation -- the single-temperature entry point delegates to the sweep when
uniforms are supplied -- so the equality holds by construction rather than by
numerical coincidence. It is pinned by a regression test at both the sampler and
the streamed-measurement level.

### Transition metrics

Per temperature and per input condition, alongside the usual concentration
statistics:

| Metric | Meaning |
|---|---|
| `tv_rank_to_greedy` | total variation between the **independently ranked** profiles of the sample and greedy. Shape only. |
| `tv_rank_to_uniform` | the same against the ranked uniform-null mean profile. Shape only. |
| `agreement_with_greedy` | fraction of positions where the sampled token **equals** the argmax token from the same logits. The one identity-preserving diagnostic. |
| `effective_support_over_null` | `N_eff(T) / N_eff(null)`; 1.0 means as broad as pure chance at the same draw count. |

The two ranked distances discard token identity before comparing, and are named
`rank` so they can never be mistaken for the same-token distances of §4. Greedy
agreement is accumulated during streaming, so it costs no extra memory.

The uniform null remains a **finite-`D` ranked occupancy reference**, not a
latent model distribution. Normalizing by it says "how concentrated is this next
to what pure chance would produce at the same draw count", nothing more.

### Fixed top-p

`top_p = 0.9` is held fixed and no top-p sweep is part of this work. The result
is therefore **not** a pure softmax-temperature experiment: raising `T` flattens
the probability profile and so changes how many ranked tokens fall inside the
0.9 nucleus. On random logits over 64 tokens the nucleus grows from about 1.9
tokens at `T = 0.12` to about 43.9 at `T = 1.2`. A `top_p = 1` control would
separate the two effects and is deliberately left outside this integration.

### Figure 5

Three standalone figures:

* `figure5_temperature_ranked_profiles.svg` — ranked real-input profiles for
  greedy, every sweep temperature, and the uniform null, on the usual log axes.
  Curves use `mean_s(sort(q_s))`: **rank within each initialization, then average
  corresponding ranks**, the same convention as figure 1;
* `figure6_temperature_ranked_distances.svg` — the two **ranked-profile** total
  variations against temperature with initialization SEM. Ranked-profile
  distances discard token identity and must not be read as the same-token TV of
  figure 2;
* `figure7_temperature_support_and_greedy_agreement.svg` — `N_eff(T)/N_eff(null)`
  per input condition on the left axis, and the fraction of positions matching
  the greedy argmax on a separate right axis. Real and shuffled often coincide
  almost exactly and are drawn with distinct widths, dashes, and markers so
  overlap reads as overlap.

Figures 5 and 6 discard token identity; the greedy-agreement curve in figure 7
is the exception.

---

## 5d. The raw predictive distribution, before any sampling policy

At every evaluation position the model emits logits. Restrict them to the
eligible support, apply softmax at temperature 1, and the result is the
probability vector these diagnostics describe:

```
p[d, i] = softmax(z[d] restricted to the eligible support)[i]
```

This vector is **upstream of every sampling decision**: no temperature scaling,
no top-p truncation, no greedy or nucleus selection. It is emphatically *not* the
nucleus distribution, which is the same logits at `T = 0.6` after top-p
truncation and renormalization.

### Four quantities that must not be confused

| Quantity | Question | Figure |
|---|---|---|
| ranked **guess frequencies** across positions | how concentrated is the aggregate distribution of what the model picks? | 1 |
| ranked **probabilities within** each position | how concentrated is one individual next-token prediction? | 8 |
| distribution of the largest per-position probability | how much mass does the greedy winner actually carry? | 9 |
| gradient magnitude vs. marginal greedy guess fraction | do the favoured tokens pull hardest? | 10 |

The first two are the pair most easily conflated, and they are independent. A
model can be nearly flat at every single position and still produce a sharply
peaked aggregate, or the reverse. Neither implies the other.

### Statistic A — ranked predictive probability profile

For each position, rank the eligible probabilities descending, then average at
**fixed rank** across positions:

```
Pbar_s(r) = mean_d p_s[d,(r)]        for each initialization s
```

The order of operations is the statistic. Ranking first and averaging second
describes the shape of a typical single predictive vector; averaging first and
ranking afterwards describes the aggregate, which is figure 1's question.

Invariants, all enforced by record validation: non-increasing in rank, finite and
non-negative, summing to 1 over the eligible support, and rank 1 equal to the
mean of the stored per-position maxima.

### Statistic B — maximum predictive probability

`p_max[d] = max_i p[d,i]`, stored per position and per initialization. It is
gathered at the greedy token rather than taken as an independent maximum, so
"the probability of the token greedy selects" is true by construction and cannot
drift from the greedy counts through a tie broken differently. Softmax is
monotone, so it is also exactly the maximum.

This separates two statements that sound alike: greedy *always* selects the
top-ranked token, but the top-ranked token need not carry much probability.

### Also persisted — target probability and loss

`p_target[d] = p[d, y_d]` and `loss[d] = -log p_target[d]`, per position and
initialization. `p_max` is confidence in the model's preferred token;
`p_target` is the mass on the actual next token. They are persisted now because
the probability vectors were already in hand, and their scientific reading
belongs with the gradient analysis.

### Storage

Only sufficient statistics. The full `[I, D, K]` probability tensor is never
written — at 12 initializations, 32768 positions and 31997 eligible tokens that
would be about 50 GiB, against roughly 12 MB for the four arrays actually kept:
`[I, K]` for the ranked profile and `[I, D]` for the three per-position vectors.

The diagnostics are computed inside the existing forward loop, from the same
softmax already used for the mean predicted mass, and only for the real input
condition. They do not touch any RNG stream, and switching them on leaves the
greedy counts, nucleus counts, sweep counts and sampling draws bit-for-bit
unchanged — which is asserted by test with the sweep both enabled and disabled.

### Temperature-conditioned greedy confidence

The same probability machinery evaluated at a grid of diagnostic temperatures:

```
T = 0.12, 0.24, 0.36, 0.48, 0.60, 1.00, 1.20      p_T[d,i] = softmax(z[d]/T)
```

**Nothing is sampled and nothing is truncated.** Temperature is used only to
reshape the logits into a probability vector whose geometry is then measured.

The invariance the analysis rests on is exact: softmax is strictly increasing, so

```
argmax_i softmax(z[d]/T)_i = argmax_i z[d,i]      for every T > 0
```

Every temperature therefore describes the **same greedy decisions**, with
different confidence attached to them. Greedy counts, token identities,
evaluation positions, logits and RNG behaviour are all unchanged; only `p_max_T`,
the ranked profile, and the entropy move.

Three facts must be kept apart:

| | changes with T? | why |
|---|---|---|
| greedy identity | **no** | argmax is scale-invariant under a positive temperature |
| confidence `p_max_T` | **yes** | the distribution sharpens as `T` falls |
| nucleus selections | **yes** | the sweep *samples* from the transformed distribution |

The third is the existing stochastic sweep and is a different experiment. These
figures are what explains why nucleus behaviour moved strongly with temperature
even though greedy token identities never did.

Per temperature the record keeps the ranked profile `[I, N_T, K]` and `p_max_T`,
`p_target_T`, `loss_T`, each `[I, N_T, D]`, plus the mean predictive entropy
`[I, N_T]`. The full `[I, N_T, D, K]` tensor is never built. Top-k cumulative
mass needs no extra array: it is a prefix sum of the ranked profile.

One sort per batch serves every temperature. Softmax is monotone, so a single
descending permutation of the logits orders every temperature's probability
vector, and the probabilities are *gathered* through it -- moving floats without
arithmetic, so the result is bit-for-bit a direct sort. That sort is computed by
the diagnostic rather than borrowed from the nucleus sweep, whose sort runs only
when sampling is enabled.

The `T = 1` slice duplicates the canonical arrays deliberately, and validation
asserts the two are identical.

Figures 11, 12 and 13 present this: the ranked profiles across temperature, a
six-panel ECDF of `p_max_T` on shared axes, and a compact confidence-versus-
temperature summary with probability and effective support on separate panels.

---

## 5e. Gradient magnitude versus guessing bias

An optional additional measurement, disabled by default, asking whether the
tokens a randomly initialized model *prefers to guess* are also the tokens whose
true-token loss pulls hardest on its parameters.

### Definition

For evaluation position `d` with true next token `y_d`:

```
z_d      = model logits at position d, restricted to the tokenizer vocabulary
z_d^elig = z_d with the structural token IDs driven to -inf
ell_d    = -log softmax(z_d^elig)[y_d]
g_d      = || grad_theta ell_d ||_2
```

`theta` is **every trainable parameter** of the model — embeddings, both decoder
blocks, the final norm, and the output projection — and the norm is the exact
global L2 norm over all of them. It is not a gradient with respect to the logits,
not one with respect to a hidden state, not the output projection alone, and not
the gradient of a window-averaged loss. No cheaper proxy stands in for it. The
per-layer diagnostic in `evaluation/gradient_norms.py` is a **different**
quantity and must not be confused with this one: it differentiates the
window-averaged loss with respect to block activations.

`ell_d` is one position's loss. The model's own `forward` returns the mean over a
window when given targets; that mean is not what is differentiated here, and
`mean_d(ell_d)` equalling it is asserted as a test.

The softmax denominator is the **eligible predictive support** — the same token
universe as the greedy and nucleus policies and as the empirical reference — so
`g_d`, `q_i` and `p_i` all speak about the same set of tokens. Structural tokens
contribute nothing to the denominator and receive exactly zero gradient.

### Aggregation

Let `S` be the set of positions the analysis covered. For token ID `i`:

```
n_i     = |{d in S : y_d = i}|                   target occurrences
G_i     = mean_{d in S : y_d = i} g_d            mean gradient norm
q_i     = |{d in S : greedy_d = i}| / |S|        greedy guess fraction
p_i     = corpus fraction over the whole analysis split
```

`G_i` is NaN, not zero, for a token that never occurs as a target in `S`: no
gradient was measured for it.

**`q_i` and `p_i` are scoped differently, on purpose.** `q_i` is computed from
the greedy predictions recorded *at the differentiated positions*, so it and
`G_i` describe the same `S`; when `S` is every evaluation position it reproduces
the run's own `greedy_counts` exactly. `p_i` deliberately stays the whole-split
corpus fraction, because it is the empirical reference the entire experiment
compares against rather than a property of the positions that were
differentiated.

### Scope

The first implementation is narrow on purpose: one initialization
(`initialization_index`, 0 by default), real input only, greedy only, no
temperature sweep, no optimizer, no parameter update, no gradient clipping. The
model is not modified in any way — parameters, buffers, existing `.grad` state,
and train/eval mode all come out as they went in, which is asserted by test.

### Cost

The measurement costs one backward pass per evaluation position, against one
forward pass per window for everything else in this protocol, so it is off unless
requested. One forward pass per window feeds `block_size` backward passes from a
retained graph, and nothing shaped `[positions, parameters]` is ever built: at
`D = 32768` over 8.6M parameters that would be about 1.1 PB. Peak memory is one
window's activations plus one gradient set.

`scripts/benchmark_position_gradients.py` measures the real cost at several
window counts before a full run is attempted.

### Persistence

The record stores the raw per-position values — `gradient_position_indices`,
`gradient_position_target_ids`, `gradient_position_greedy_ids`, and
`gradient_position_norms`, each `[D_g]` — rather than pre-aggregated per-token
sums. `G_i`, `n_i` and `q_i` are derived from them by
`analysis/gradients.py`, so a later re-aggregation never requires recomputing a
gradient.

### Temperature-conditioned gradients

The same grid of temperatures, but now **inside the loss** rather than inside a
diagnostic probability:

```
ell_T(d) = -log softmax(z_d / T)[y_d]      g_d(T) = || grad_theta ell_T(d) ||_2
d ell_T / d z_i = (p_T(i) - 1[i = y]) / T
```

Both routes are intended: the explicit `1/T` and the sharpening of `p_T`. **No
compensating `T^2` factor** is applied — this is plain temperature-scaled cross
entropy, not a distillation loss.

This is a controlled experiment because of the invariance. Softmax is strictly
increasing, so `argmax softmax(z/T) = argmax z`, and the greedy guess fraction
`q_i` is *exactly* the same at every temperature. So is `n_i`, which counts
targets, and so is `p_i`. Anything that moves across the panels of figure 10
moves for gradient reasons alone.

Two limits, both analytic and both tested:

* the greedy winner is the target → `p_T(y) → 1`, and the gradient → 0 as
  `T → 0`, because the probability error decays exponentially in `1/T` and beats
  the explicit `1/T`;
* the greedy winner is not the target → `|p_T - onehot| → sqrt(2)`, so the logit
  gradient grows as `O(1/T)`.

One forward graph per window is retained and re-differentiated once per position
*per temperature*, so no extra forward pass is taken and no
`[positions, temperatures, parameters]` tensor is ever formed. `T = 1` is one
row of that grid, not a second implementation: dividing by exactly 1.0 is the
identity in IEEE-754, so the canonical row is bitwise the established observable,
and validation asserts it equals the stored canonical array.

### Figure 10

`figure10_temperature_gradient_vs_initial_guess_bias.svg`: six panels, one per
sweep temperature, each a token scatter of `G_i(T)` against the *fixed* `q_i`,
coloured by `p_i` on one shared logarithmic colour scale with a single colorbar.

**The y axis is a count transform, shared by all six panels.** Since
`q_i = k_i / D` with `k_i` an integer guess count, the axis plots `log10(1 + k_i)`:
`q = 0` maps to exactly 0, a full 0.30 of height separates "never guessed" from
"guessed once", the low counts where nearly all tokens live are expanded, and the
tail is compressed. It is monotone and exactly invertible, so it reorders nothing
and discards nothing, and the ticks are labelled `k/D` so the axis still reads as
a fraction. This replaced symlog, which kept the zeros but still spent most of the
height on the few tokens with large `k`: at the real scale about 94% of tokens
have `q_i = 0` and formed one indistinguishable line.

**The x limits are per panel.** `G_i(T)` sits at a different place for every
temperature — the explicit `1/T` alone moves the median by an order of magnitude
across the grid — so shared limits leave every cloud in a sliver of its panel. The
transformation stays logarithmic in all six; only the limits differ, each covering
its own full range with modest padding so nothing is clipped. Every panel states
its interval in its title, and the figure says in words that horizontal position
is **not** comparable between panels.

The single-temperature scatter remains available as
`supplementary_t1_gradient_vs_initial_guess_bias.svg`, and older records that
carry only the canonical slice still render it.

### Legacy single-temperature figure

One marker per token with
`n_i > 0`, at `x = G_i` and `y = q_i`, coloured by `p_i`.

Three scale decisions are made from the observed distributions rather than by
habit, and all three are consequences of what the data actually looks like.

The **y axis is symmetric-log with its linear region ending at one guess**,
`1/D`. The overwhelming majority of tokens are never greedily guessed, and
`q_i = 0` is a measured outcome rather than missing data. A plain logarithmic
axis would delete exactly the tokens the figure exists to display. Because `q_i`
is a multiple of `1/D`, the only attainable value below the threshold is zero, so
the linear region is precisely the gap between "never guessed" and "guessed
once"; a reference line marks it.

The **x axis follows the observed dynamic range** of `G_i`: logarithmic once it
spans at least a decade, linear otherwise, since a log axis over a half-decade
spreads noise and hides structure. The choice is printed in the axis label.

The **colour is logarithmic** over positive `p_i`, using a perceptually uniform
map. Corpus frequency is heavy-tailed; a linear map would collapse everything but
the few most common tokens into one shade. Nothing is clipped — the normalization
spans the real minimum and maximum.

Every plotted token necessarily has `p_i > 0`, because an evaluation target is a
position in the split. The figure still guards the case and states any omission
rather than silently losing a marker.

### Correlations

Reported beside the figure, over the same tokens it plots:

* Spearman `rho(G_i, q_i)` — the primary quantity;
* Spearman `rho(G_i, p_i)`;
* Spearman `rho(q_i, p_i)`.

Rank correlation is tie-corrected with average ranks. That is not a refinement:
nearly all tokens share the single tie at `q_i = 0`, and ordinal ranking would
impose an arbitrary order inside that block and report a coefficient partly
manufactured by the sort. A constant input yields NaN, not zero, because the
coefficient is undefined rather than absent.

Tokens with `q_i = 0` are **included**. Dropping them would bias the primary
coefficient toward the tokens the model already favours.

The primary coefficient is deliberately **unweighted**, although `G_i` is a mean
over `n_i` positions and is far noisier for a token seen once than for one seen a
hundred times. Weighting answers a different question and would need its own
justification, so the imprecision is reported instead — as a stratification by
`n_i` band and a sensitivity sweep over `n_i >= k` — and left visible.

---

## 6. The figures

| Figure | Question | Token identity |
|---|---|---|
| 0 — sampling adequacy | Is `p_selected` representative of `p`? | — (corpus only) |
| 1 — ranked concentration | How concentrated are the distributions, after ranking each independently? | **discarded** |
| 2 — same-token mismatch | How large are same-token discrepancies, and how much survives averaging over seeds? | **preserved** |
| 3 — token identity | For the same canonical token, how does mean guess frequency compare with corpus frequency? | **preserved** |

Figures 0 and 1 describe **shape and concentration**. Figures 2 and 3 describe
**token-identity alignment**. Two distributions can match perfectly in shape
while disagreeing completely about which tokens carry the mass, so the two
families answer genuinely different questions and neither implies the other.

In figure 1, rank `r` of the guess curve and rank `r` of the corpus curve are
generally **different tokens**. Nothing about token agreement may be read from
it.

---

## 7. Observations so far

Everything below is **validation and pilot observation**, not a final result.
Sample sizes were chosen to exercise the machinery, not to support a scientific
claim.

### 7.1 Character baseline

Tiny tracked fixture, character tokenizer. `N`=256, `I`=3, `R`=2.

| Quantity | Value |
|---|---|
| `V` full / eligible / corpus-observed | 39 / 39 / 36 |
| corpus `N_eff` | 21.28 |
| train split | 476 tokens |
| selected `N_eff` | 20.38 |
| TV(split, selected) | 0.072216 |
| JS(split, selected) | 0.009000 |
| greedy `N_eff` | 17.099 ± 1.561 |
| nucleus `N_eff` | 35.173 ± 0.201 |
| greedy zero-frequency | 11.67 ± 2.67 (29.91%, `N`=256) |
| nucleus zero-frequency, per replicate | 0.00 ± 0.00 (0.00%, `N`=256) |
| pooled nucleus coverage | 0.0 unreached over 512 pooled draws |
| greedy TV from corpus | 0.588771 ± 0.003792 |
| nucleus TV from corpus | 0.464740 ± 0.016905 |
| runtime / peak RSS | 0.032 s / 515.3 MiB |
| model parameters | 459,392 |

Sampling adequacy is good here: with 256 positions over a 39-token vocabulary,
the selected targets track the split closely.

### 7.2 Realistic subword tokenizer

`mistralai/Mistral-7B-v0.1`, tokenizer artifacts only, loaded offline from cache
after a single acquisition. Cache footprint ≈ 2.3 MB; no model weight files
present. **Revision is currently unpinned.**

| Quantity | Value |
|---|---|
| `V` full | 32,000 |
| `V` eligible | 31,997 (excludes BOS=1, EOS=2, UNK=0) |
| `V` corpus-observed | 6,181 |
| corpus `N_eff` | 768.94 |
| train split | 45,549 subword tokens |

The gap between 6,181 observed and 768.94 effective is the point: the corpus
touches about a fifth of the vocabulary, but its mass behaves like roughly 769
equally likely tokens.

### 7.3 Subword smoke validation — `N`=512

Deliberately small; **validation only**. `I`=2, **`R`=2 (historical protocol)**.

| Quantity | Value |
|---|---|
| selected `N_eff` | 176.11 |
| TV(split, selected) | 0.547250 |
| JS(split, selected) | 0.293443 |
| greedy `N_eff` | 213.514 ± 12.888 |
| nucleus `N_eff` | 998.424 ± 3.634 |
| greedy zero-frequency | 31,682.00 ± 4.00 (99.02%, `N`=512) |
| nucleus zero-frequency, per replicate | 31,489.50 ± 1.00 (98.41%, `N`=512) |
| pooled nucleus coverage | 30,991.5 unreached over 1,024 pooled draws |
| greedy TV from corpus | 0.992767 ± 0.000450 |
| nucleus TV from corpus | 0.979885 ± 0.000532 |
| runtime / peak RSS | 0.894 s / 1,082.5 MiB |

TV(split, selected) = 0.55 is far too large for any model comparison built on
these positions to be trusted. That is exactly what figure 0 exists to reveal.

### 7.4 Subword pilot — `N`=8192

`I`=4, **`R`=4 (historical protocol)**. **Pilot, not the serious experiment.**

These values were measured under `R = 4` and are kept exactly as observed. Do not
restate them as though they used `R = 1`, and do not compare their nucleus SEMs
naively with future `R = 1` runs.

| Quantity | Value |
|---|---|
| selected `N_eff` | 558.63 |
| TV(split, selected) | 0.219879 |
| JS(split, selected) | 0.082817 |
| greedy `N_eff` | 810.674 ± 22.763 |
| nucleus `N_eff` | 18,107.174 ± 36.467 |
| greedy zero-frequency | 29,183.00 ± 19.00 (91.21%, `N`=8192) |
| nucleus zero-frequency, per replicate | 24,786.06 ± 0.74 (77.46%, `N`=8192) |
| pooled nucleus coverage | 11,610.8 unreached over 32,768 pooled draws |
| greedy TV from corpus | 0.966604 ± 0.001243 |
| nucleus TV from corpus | 0.865343 ± 0.000869 |
| runtime | ≈73 s internal / ≈79.65 s external wall clock |
| peak RSS | ≈1.17 GiB |
| model parameters | 8,585,856 |

Observations, stated as observations:

- Adequacy improved substantially from `N`=512 (TV 0.547 → 0.220) but the
  selected distribution still differs materially from the split, and selected
  `N_eff` (558.63) remains well below corpus `N_eff` (768.94). **8,192 positions
  are not yet an accepted evaluation sample.**
- The equal-draw correction matters at this scale: the comparable nucleus figure
  is 24,786 tokens unreached, while the pooled figure over four times as many
  draws is 11,611. Reporting the latter beside greedy's 29,183 would have
  overstated nucleus coverage by more than a factor of two.
- Both policies sit far from the corpus distribution (TV 0.87–0.97), which is
  unremarkable for an untrained model and is not a quality statement.

---

## 8. Reproducing

Character baseline:

```bash
python3 scripts/run_initialization_distribution_experiment.py \
  --data-config configs/data/tiny_text.yaml \
  --num-initializations 3 --num-windows 16 --block-size 16 --num-replicates 2
```

Realistic subword tokenizer (after `pip install -e ".[tokenizers]" ".[analysis]"`):

```bash
python3 scripts/run_initialization_distribution_experiment.py \
  --data-config configs/data/wikitext2_subword.yaml \
  --model-config configs/model/tiny_llama_32k.yaml \
  --offline --forward-batch-size 4
```

These are the settings used above. They are **pilot settings**, not recommended
scientific settings.

Results land under `outputs/`, which Git ignores. Run directories are named
interpretably — see [`README.md`](../README.md) — and hold `analyses/` (the
complete per-token record plus a scalar summary), `figures/` (four SVGs),
`config/` (verbatim snapshots), `metrics/`, and `metadata.json`.

Read a record with `notebooks/initialization_distribution.ipynb`.

---

## 9. Open items before the serious experiment

0. **Review and accept the null-model integration** (uniform output null,
   input-structure conditions, `R = 1`) before anything below is decided on top
   of it.
1. **Pin the tokenizer revision.** Currently `revision: null`. The tokenizer now
   determines the vocabulary, the token stream, and therefore every distribution
   being compared, so an unpinned revision undermines reproducibility more than
   an unpinned dataset revision would.
2. **Calibrate the evaluation-position count.** `N`=8192 still leaves
   TV(split, selected) ≈ 0.22. Increase `N` until figure 0 shows the selected
   distribution tracking the split, judging against corpus `N_eff` ≈ 769 rather
   than against `V` = 32,000.
3. **Choose `I` and `R` from observed spread.** The pilot SEMs are the evidence;
   the character-run values do not transfer, because vocabulary, support, and
   frequency structure all changed at once.

Runtime scales roughly linearly in `N`: ≈73 s at `N`=8192 with `I`=4, `R`=4, at
about 1.17 GiB peak. That is the budget any larger choice has to fit.

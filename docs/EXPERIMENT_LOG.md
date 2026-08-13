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

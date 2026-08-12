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
                 R controlled replicates
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
3. **Token sampling** — affects only the nucleus policy, driven by its own seed,
   averaged over replicates *within* an initialization before initializations are
   compared.

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

## 5. The figures

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

## 6. Observations so far

Everything below is **validation and pilot observation**, not a final result.
Sample sizes were chosen to exercise the machinery, not to support a scientific
claim.

### 6.1 Character baseline

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

### 6.2 Realistic subword tokenizer

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

### 6.3 Subword smoke validation — `N`=512

Deliberately small; **validation only**. `I`=2, `R`=2.

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

### 6.4 Subword pilot — `N`=8192

`I`=4, `R`=4. **Pilot, not the serious experiment.**

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

## 7. Reproducing

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

## 8. Open items before the serious experiment

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

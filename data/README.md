# Data subsystem

This folder contains local dataset material used by the LLM Behavior Lab project.

The project separates data handling from models, inference, and evaluation so that datasets can be changed without modifying model code. This separation is important for later phases, where the same model and analysis utilities will be reused across different dataset scales, dataset sources, training runs, checkpoints, and fine-tuning datasets.

At the current stage, the data subsystem supports:

- a small local text corpus for smoke tests and lightweight analysis
- config-driven dataset selection
- optional Hugging Face dataset loading for standard language-modeling datasets
- deterministic character-level tokenization
- train/validation token splitting
- causal language-modeling batch creation

The data pipeline is intentionally simple for now. It is designed to be easy to inspect, easy to test, and easy to extend before larger-scale training infrastructure is added.

---

## Folder role

The `data/` folder is the place for local raw or downloaded datasets.

The source code that loads and processes data lives under:

```text
src/llm_behavior_lab/data/
```

The dataset configuration files live under:

```text
configs/data/
```

This split is intentional:

```text
data/
  raw or downloaded data files

configs/data/
  YAML files describing which dataset source to use

src/llm_behavior_lab/data/
  Python utilities for loading, tokenizing, splitting, and batching text
```

This keeps raw data separate from reusable data-loading logic.

---

## Git tracking policy

Dataset contents under `data/` are ignored by Git, while Markdown documentation inside the folder stays tracked:

```gitignore
data/**
!data/
!data/**/
!data/**/*.md
```

The `!data/` and `!data/**/` lines re-include directories so Git can still see allowed files inside them. The `!data/**/*.md` line is why this guide remains tracked.

This matters because local or downloaded corpora may be large. Anything you place under `data/` is ignored by default, with no further configuration needed.

Dataset configuration files remain tracked under:

```text
configs/data/
```

Two files under `data/` are tracked on purpose:

| Path | Why it is tracked |
|---|---|
| `data/README.md` | This guide, kept by the Markdown re-include rule. |
| `data/raw/tiny_corpus.txt` | The small offline fixture used by the default configs, the documented script commands, and the test suite. It is committed deliberately so the repository stays runnable and testable without any download. |

Everything else — downloaded datasets, generated dataset artifacts, and cached local data — should live under `data/` and stay out of Git.

Do not bulk-untrack the folder with `git rm -r --cached data`. That would remove both the tiny corpus and this guide from the repository, and the ignore rules would then prevent the corpus from being added back. If a specific large file was committed by mistake, untrack that one path deliberately instead.

---

## Current datasets and source types

A dataset is chosen entirely through configuration. The `dataset.source` field
selects how it is obtained; everything downstream — tokenization, splitting,
batching — is identical either way.

| Source | Meaning |
|---|---|
| `local_text` | A text file already on disk. The default, and the only one that needs no optional dependency. |
| `huggingface` | A dataset obtained from the Hugging Face Hub and prepared to disk. |

### 1. Local tiny text corpus

The default lightweight dataset is the small corpus tracked in this repository:

```text
data/raw/tiny_corpus.txt
```

Config: `configs/data/tiny_text.yaml`. It is used for fast smoke tests,
tokenizer checks, data/model compatibility checks, inference checks, the
initialization analysis, and gradient-norm diagnostics.

It is not meant for meaningful training. Its purpose is to keep a fresh clone
runnable and testable with no setup and no downloads.

```yaml
dataset:
  name: tiny_local_text
  source: local_text        # optional; this is the default
  path: data/raw/tiny_corpus.txt
  val_fraction: 0.2
```

A local dataset resolves from its configured path and never consults the data
root, a cache, or the network.

### 2. Hugging Face datasets

External datasets need the optional dependency:

```bash
python3 -m pip install -e ".[hf]"
```

Shipped configs: `configs/data/wikitext2.yaml` and
`configs/data/tinystories.yaml`. Both cap the number of examples and characters
so the prepared corpus stays small.

```yaml
dataset:
  name: wikitext2_raw_train1k
  source: huggingface
  repo_id: Salesforce/wikitext
  subset: wikitext-2-raw-v1
  split: train[:1000]
  revision: null            # pin for reproducibility
  text_field: text
  max_examples: 1000
  max_characters: 200000
  document_separator: "\n\n"
  val_fraction: 0.1
  verify: false
```

The first run downloads the dataset, announces what it is doing, and writes a
prepared copy under the data root. Later runs read that copy without any network
access. The contents are never committed.

The character tokenizer derives its vocabulary from the prepared text, so model
compatibility depends on the corpus, its limits, and its upstream revision. Both
shipped external configs were live-tested end to end against
`configs/model/tiny_llama.yaml` at their current limits — WikiText-2 gave a
tokenizer vocabulary of 147 and TinyStories 70, both under `vocab_size: 256`.
Those are observations, not guarantees. If a run reports that the tokenizer
vocab size exceeds the model vocab size, raise `model.params.vocab_size`.

---

## Where datasets are stored

Dataset **code and configuration** are tracked. Dataset **contents** are not.

Downloaded and prepared data live under an external data root, chosen in this
order:

1. `--data-root` on the command line
2. the `LLM_BEHAVIOR_LAB_DATA_ROOT` environment variable
3. `$XDG_CACHE_HOME/llm-behavior-lab`
4. `~/.cache/llm-behavior-lab`

The default is always outside the repository, so a large download cannot land in
a tracked path. On a cluster, set the environment variable to a scratch
location:

```bash
export LLM_BEHAVIOR_LAB_DATA_ROOT=/scratch/$USER/llm-behavior-lab
```

The root is laid out as:

```text
<data_root>/
  prepared/<dataset-name>/
    text.txt          the corpus, ready to tokenize
    manifest.json     what it is and how it was produced
  hf/                 Hugging Face cache
```

A prepared copy inside the repository at `data/prepared/<name>/` is read when it
exists, which is convenient on a single machine. It is never the destination for
new data.

---

## How a dataset is found

Every part of the project obtains a dataset through one function,
`resolve_dataset`, which tries these locations in order:

1. the configured `dataset.path`
2. a prepared copy in the repository, `data/prepared/<name>/`
3. a prepared copy under the data root
4. a copy already in the Hugging Face cache, used without any network
5. acquisition, when permitted and after announcing it

If none succeeds, the error names every location tried, says why acquisition did
not happen, and gives the command that would fix it.

### Limitations to know about

**Replacement is not transactional.** An existing prepared copy is removed
before the new one is renamed into place. A crash in that window leaves neither,
and the next run prepares again. The guarantee that does hold unconditionally is
that an incomplete preparation is never mistaken for a complete one.

**Loader calls are serialized within a process.** Enabling offline mode means
changing process-global state, so Hugging Face loader calls in this package run
one at a time. A cache-only call therefore cannot force a concurrent
acquisition offline, and overlapping calls cannot leave the process
permanently offline. This covers threads inside one process only.

**Concurrent preparation is not supported.** Two processes preparing the same
dataset under the same data root may collide at replacement time and one will
fail with a filesystem error. Nothing deletes another process's staging
directory, and no observer ever sees a half-written dataset, so the failure is
safe and a re-run fixes it. Abandoned `.partial` directories are inert and
ignored; remove them by hand if they accumulate.

**Offline mode is enforced by Hugging Face, not by this project.** Cache reads
run with `HF_HUB_OFFLINE` set and with the corresponding library constants set,
because both libraries read that variable once at import. There is no
independent network barrier here: the guarantee is only as strong as the
library honouring its own offline flag.

**`--offline` and `--no-download` currently follow the same code path.** Both
read from the local cache and neither can fetch. They differ only in intent and
in what the error message says.

**Upstream drift is not detected.** With `revision` unpinned, a prepared copy
stays valid even if the upstream dataset changes. Pin `revision` when that
matters, using a real Hub reference obtained from the dataset repository (see
*Four identifiers* below) — the manifest does not supply one.

**Streamed datasets are released explicitly.** Preparing from a streaming
dataset consumes only part of it, and a partially consumed stream can hold
native resources that are otherwise freed at interpreter shutdown. In some
dependency combinations that late release aborts the process with `SIGABRT`
(exit status 134) *after* preparation has already finished and the corpus has
been published — a correct run that looks like a failed one.

What was established, and its limits:

- The abort reproduces with a few lines of `datasets` alone, with no code from
  this project involved, so it is not caused by the resolver or by preparation.
- Dropping ordinary Python references to the dataset is **not** sufficient.
- Running an explicit `gc.collect()` before shutdown **is** sufficient: the same
  reproduction then exits 0.
- Disabling the Xet transfer backend does **not** avoid it, so the responsible
  component is not identified. No claim is made here about which compiled
  dependency owns the resource.

`materialize` therefore drops its reference to a streamed dataset and collects
as soon as nothing needs it — after provenance is read and the corpus is built,
before the prepared directory is written — including when preparation fails
part-way. Ordinary non-streaming loads are untouched: they do not have the
problem, and forcing collection is not free. This is a scoped mitigation for
observed behavior, not a fix for the underlying defect, and it is expected to
become unnecessary once that defect is resolved upstream. One residual case
stays outside its reach: while an exception is still propagating, the traceback
keeps the frames that referenced the dataset alive, so collection cannot free it
until the exception is discarded.

### Controlling acquisition

Every data-consuming script accepts the same options:

| Option | Effect |
|---|---|
| `--data-root PATH` | Where prepared and cached data live. |
| `--no-download` | Reuse what is already available; never obtain anything missing. |
| `--offline` | Forbid all network access. Implies `--no-download`. |
| `--force-refresh` | Re-acquire, replacing an existing prepared copy. See below. |

Scripts acquire missing data by default, and always announce it first. Code that
imports the library gets the conservative default instead: a
`ResolutionPolicy()` built without arguments can never reach the network.

#### Measuring what an acquisition actually costs

The prepared corpus is bounded by `max_examples` and `max_characters` —
`max_characters` bounds the final file, separators included, so
`num_characters <= max_characters` always holds in the manifest. Streaming means
the whole split need not be materialized before those limits apply.

In live validation this worked as intended: preparing TinyStories produced a
168 KB corpus and left a 13 MB shared cache, against source metadata describing
a roughly 1 GB download. What it cost over the network was **not** measured.

`/usr/bin/time -v` reports process resource use — wall time, peak memory, and
block-device I/O — and `du -sh` on the data root reports local disk footprint,
so a before/after pair gives data-root growth. Neither observes sockets, so
neither measures bytes transferred. For streamed datasets `du` is not even a
proxy, because streamed shards need not land in the cache at all.

Exact network transfer remains unmeasured unless a network-monitoring method is
used alongside a real acquisition.

#### What `--force-refresh` guarantees

Prepared and cache-only reuse are bypassed, the source is opened again, and
whatever it yields becomes a replacement prepared corpus. The resulting bytes
are **not** guaranteed to differ: upstream may legitimately be unchanged.

For non-streaming datasets the loader is additionally asked to re-fetch rather
than reuse its own download cache. For streamed datasets that request has no
meaningful effect, because streaming bypasses the download-and-prepare step it
governs; there the guarantee is simply that the current source stream is opened
again.

A refresh that cannot reach the source fails and leaves the previous prepared
copy intact.

### Detecting stale or incomplete data

`manifest.json` records the identity that produced the corpus — source, subset,
split, revision, text field, and limits — plus a SHA-256 digest and the
character and document counts. It holds no filesystem paths, so a prepared
directory can be moved between machines.

The manifest is written last, inside a staging directory that is then moved into
place. Its presence therefore means the preparation finished; a directory
without one is an interrupted attempt and is rebuilt rather than trusted, so an
incomplete new preparation is never mistaken for a complete one. Replacing an
*existing* prepared copy is a separate matter and is not transactional — see
*Limitations to know about* above.

The manifest also records what the source reported about itself — fingerprint,
version, config name, split, and sizes — separately from what was requested, so
a later reader can tell which corpus a run actually used. Those resolved values
are excluded from staleness comparison, because two preparations of the same
request may legitimately differ there.

#### `download_size` and `dataset_size` are source metadata

These two fields come from the dataset's own published metadata and describe the
**whole published dataset or config**. They are not measurements of this
acquisition: not bytes transferred, not local cache growth, and not the size of
the prepared corpus.

The live TinyStories run makes the gap concrete. Its manifest reports roughly a
1 GB `download_size` and a 2 GB `dataset_size`, while the entire data root after
preparing both external datasets was about 14 MB — a 168 KB prepared corpus, a
208 KB WikiText corpus, and a 13 MB shared Hugging Face cache. Read these fields
as "how large the upstream dataset is", never as "what this run cost".

That 14 MB is a disk measurement. It does not establish how many bytes crossed
the network.

#### Four identifiers, easily confused

| Identifier | What it is | Where |
|---|---|---|
| Requested Hub revision | A branch, tag, or commit SHA sent to the loader. Empty unless you pin it. | `dataset.revision`, copied to the manifest |
| Datasets fingerprint | The Datasets library's **local** cache/state hash. Not a Hub reference and not accepted by `revision`. Absent for streamed datasets. | `manifest.resolved.fingerprint` |
| Prepared-text SHA256 | Digest of the exact `text.txt` this project wrote. The reliable answer to "did two runs use the same corpus?" | `manifest.sha256` |
| Actual Hub commit | The commit that served the data. **Not recorded**; obtaining it would need a Hub lookup, which the resolver deliberately avoids. | nowhere |

To pin strictly, take a branch, tag, or commit SHA from the dataset's page on
huggingface.co (or from `huggingface_hub`) and set `dataset.revision` to it.
Nothing in a prepared manifest can be pasted into that field.

If the configuration later disagrees with the manifest — a different split or a
larger `max_characters` — resolution reports the differing field instead of
silently reusing the old corpus. Setting `verify: true` additionally re-checks
the digest on every load.

---

## Data-loading code

```text
src/llm_behavior_lab/data/
  config.py        DatasetConfig, ResolutionPolicy, resolve_data_root
  errors.py        the dataset error hierarchy
  resolver.py      resolve_dataset, ResolvedDataset
  prepared.py      prepared directories and manifests
  huggingface.py   acquisition (the only module that can reach the network)
  cli.py           the shared command-line options
  text_dataset.py  load_text_file, split_token_ids
  tokenizer.py     CharTokenizer
  dataloader.py    CausalLMBatch, CausalLMBatcher
```

The entry point returns a small object describing what was found:

```python
resolved = resolve_dataset(dataset_config, policy, repo_root=REPO_ROOT)
text = resolved.read_text()
```

`ResolvedDataset` carries the dataset name, source, path, the route that
supplied it, and provenance for experiment metadata. It exposes a path rather
than a string of text, so later phases can stream large corpora without changing
the interface.

---

## Data-loading flow

```text
data config
   |
   v
DatasetConfig.from_config          identity: which dataset
   |
   +-- ResolutionPolicy            policy: what this invocation may do
   |
   v
resolve_dataset                    local first, acquire only if permitted
   |
   v
ResolvedDataset.read_text()
   |
   v
CharTokenizer.from_text -> encode -> split_token_ids -> CausalLMBatcher
   |
   v
LLaMA-style model
```

Identity and policy are kept apart on purpose. Identity belongs in tracked YAML
and is snapshotted with a run. Policy comes from flags and the environment, is
machine-specific, and never appears in a config file — which is why the data
root is not a config field.

---

## Scripts using the data pipeline

| Script | Purpose |
|---|---|
| `prepare_dataset.py` | Stage a dataset ahead of time, or report what is missing. |
| `check_data_pipeline.py` | Text to batches, with an optional model forward pass. |
| `run_inference.py` | Build the tokenizer from the dataset and run inference. |
| `analyze_untrained_model.py` | Initialization analysis, optionally persisted. |

Prepare a dataset before using it, which is useful on a login node before a
batch job:

```bash
python3 scripts/prepare_dataset.py --data-config configs/data/wikitext2.yaml
```

Check what is available without obtaining anything:

```bash
python3 scripts/prepare_dataset.py \
  --data-config configs/data/wikitext2.yaml \
  --no-download
```

Check the local pipeline:

```bash
python3 scripts/check_data_pipeline.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml
```

Check dataset loading only, skipping the model:

```bash
python3 scripts/check_data_pipeline.py \
  --data-config configs/data/tiny_text.yaml \
  --model-config configs/model/tiny_llama.yaml \
  --skip-model-check
```

Run an experiment workflow on an external dataset. It is prepared on first use
and reused afterwards:

```bash
python3 scripts/analyze_untrained_model.py \
  --data-config configs/data/wikitext2.yaml \
  --model-config configs/model/tiny_llama.yaml
```

---

## Practical limitations

The current data subsystem is intentionally simple.

Current limitations:

- the character tokenizer is rebuilt from the loaded text each run
- tokenizers are not yet saved or reused
- external datasets are concatenated into one text stream
- there is no persistent tokenized cache yet; preparation stores plain text
- streaming/sharded training loaders are not implemented yet
- there is no train/validation/test split metadata file yet
- large-scale training is not implemented yet
- dataset statistics are not yet saved as structured artifacts

These limitations are acceptable for the current phase because the project is still building stable foundations before training infrastructure.

---

## Future data extensions

Planned data-side extensions include:

- larger language-modeling datasets
- persistent tokenizer artifacts
- persistent tokenized datasets
- streaming dataset support for training
- dataset sharding
- train/validation/test split improvements
- dataset statistics and cached summaries
- token grouping for bias and preference analysis
- subgroup-aware or metadata-aware datasets
- supervised fine-tuning datasets
- instruction-tuning datasets
- checkpoint-compatible dataset state for reproducible training runs

These should be added gradually so the data path remains easy to test and debug.

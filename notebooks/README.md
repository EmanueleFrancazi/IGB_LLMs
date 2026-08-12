# Notebooks

Readable scientific logs for experiments whose results already exist.

## Folder role

A notebook here is a **reading surface**, not an implementation. It loads a persisted
experiment record, calls functions from `llm_behavior_lab.analysis`, and explains what the
numbers mean. Computation stays in the package, where it can be tested and reused; if a
cell needed more than a few lines of logic, that logic belongs in
`llm_behavior_lab.analysis.aggregation` instead.

This split is what keeps the notebooks worth revisiting. A cell that silently reimplements
a statistic can disagree with the tested version, and nobody notices until a figure is
wrong.

## Contents

| Notebook | Question it answers |
|---|---|
| `initialization_distribution.ipynb` | At random initialization, how do the model's selected token guesses compare with the corpus token distribution, and how stable is that across initializations? |

## Running one

Notebooks read results; they do not produce them. Run the experiment first:

```bash
# smoke check on the tracked fixture: fast, and not scientifically meaningful
python3 scripts/run_initialization_distribution_experiment.py \
  --data-config configs/data/tiny_text.yaml \
  --num-initializations 3 --num-windows 16 --block-size 16 --num-replicates 2

# first meaningful experiment
python3 scripts/run_initialization_distribution_experiment.py \
  --data-config configs/data/wikitext2.yaml --offline
```

Then open the notebook and point `RUN_DIR` at the run you want to read. Figures are
regenerated from the record, so a notebook never depends on files produced by an earlier
session.

Plotting needs the optional extra:

```bash
python3 -m pip install -e ".[analysis]"
```

## Git policy

Notebooks are tracked; their **outputs are not**. Commit with cleared outputs so the diff
stays reviewable and the repository stays small. Everything a reader needs to reproduce a
figure is in the record under `outputs/`, which is ignored by Git, plus the command above.

Clear outputs before committing:

```bash
jupyter nbconvert --clear-output --inplace notebooks/initialization_distribution.ipynb
```

`.ipynb_checkpoints/` is already ignored.

## Adding a notebook

1. Put the measurement in `llm_behavior_lab.analysis.aggregation` and test it.
2. Put the figure in `llm_behavior_lab.analysis.figures`, drawing only what aggregation
   returns.
3. Add a notebook that loads a record, calls both, and explains the result.
4. Keep the section headings close to the existing ones — question, setup, protocol, then
   one section per figure, then observations and caveats. The repetition is deliberate:
   experiments become comparable when their write-ups have the same shape.

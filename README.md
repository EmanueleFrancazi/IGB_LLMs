# LLM Behavior Lab

This repository is an incremental research codebase for studying how language-model behavior evolves from random initialization through pre-training and fine-tuning.

The project prioritizes:

- transparent model implementations
- modular data, training, inference, evaluation, and logging components
- reproducible experiments
- easy extension to additional model families

## Current phase

Phase 1 creates the repository skeleton and the core interfaces needed before integrating the first real model implementation.

This phase includes:

- package structure under `src/llm_behavior_lab/`
- a base language-model interface
- a simple model registry
- reproducibility and device utilities
- a tiny debug language model used only for smoke tests
- a YAML model config
- a smoke-test script
- basic pytest checks

The debug model is not the research baseline. It exists only to verify that the package, registry, config loading, forward pass, loss computation, and parameter counting work before Phase 2 model integration.

## Install

From the repository root:

```bash
python -m pip install -e ".[dev]"
```

## Run the smoke test

```bash
python scripts/smoke_test_model.py --config configs/model/tiny_debug.yaml
```

Expected output includes:

- selected device
- model name
- parameter count
- input, target, and logits shapes
- scalar loss value

## Run tests

```bash
pytest
```

## Next phase

Phase 2 will integrate the selected LLaMA-style educational implementation with minimal architecture changes, while preserving the registry and model interface introduced here.

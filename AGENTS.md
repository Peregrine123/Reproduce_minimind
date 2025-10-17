# Repository Guidelines

## Project Structure & Module Organization
- `main.py` is the shared CLI for pretrain/SFT and auto-switches to DDP when multiple GPUs are present.
- `model/` hosts `model_minimind.py` and tokenizer assets; treat it as the canonical source of configs and hyperparameters.
- `triainer/` (intentional spelling) provides `train_pretrian.py` and `train_full_sft.py` for direct launches or torchrun jobs.
- `dataset/` bundles `lm_dataset.py` plus JSONL corpora; route generated checkpoints and logs to `out/`.

## Build, Test, and Development Commands
- `uv sync` installs dependencies from `pyproject.toml`, including the pinned CUDA wheel index.
- `python main.py --mode pretrain --epochs 2 --batch_size 32` runs the default single-node pretraining loop (wandb optional).
- `python triainer/train_full_sft.py --data_path dataset/sft_data.jsonl` triggers SFT without the orchestration helpers.
- `torchrun --nproc_per_node=2 main.py --mode pretrain --ddp` reproduces distributed training; checkpoints land in `out/`.

## Coding Style & Naming Conventions
- Target Python 3.11 with four-space indentation, and add typing when extending utilities around `MiniMindConfig`.
- `ruff check` and `ruff format` enforce the 88-character, double-quote style; `model` and `triainer` remain first-party imports.
- Use snake_case for functions/variables, PascalCase for classes, and keep CLI flags in sync with docstrings.
- Keep current spellings (`triainer`, `PretrianDataset`) so relative imports continue to resolve.

## Testing Guidelines
- Run `python test_gpu_detection.py` before multi-GPU experiments to confirm CUDA readiness.
- Smoke-test data changes with truncated JSONL slices and overrides like `--batch_size 2 --max_length 128`.
- Seed `torch`, `random`, and `numpy` when comparing losses to keep results reproducible.
- Store new validation scripts under `tests/` or `tools/` and mention them in README updates.

## Commit & Pull Request Guidelines
- Follow the concise, verb-led commit tone already in history (e.g., `修复MoE前向传播中的张量形状错误`).
- Keep pull requests focused: call out affected subsystems, list validation commands, and link issues or experiment IDs.
- Attach wandb run URLs or screenshots for training tweaks, and flag external assets reviewers must fetch.

## Data & Configuration Tips
- Store large artifacts outside git; document expected mounts (e.g., datasets under `./dataset`) and skip binary commits.
- Surface new config knobs via `argparse` in `main.py` and mirror defaults in `triainer/` scripts to retain parity.

## 沟通准则
- 默认使用中文回答所有问题，除非用户明确要求切换到其他语言。

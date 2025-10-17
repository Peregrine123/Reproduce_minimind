# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

这是一个 MiniMind LLM 的复现项目，包含了一个基于 Transformer 架构的小型语言模型实现。项目支持标准的 Transformer 结构以及混合专家（MoE）模式。

## Environment Setup

项目使用 uv 作为包管理器，配置在 pyproject.toml 中：

```bash
# 创建虚拟环境并安装依赖
uv sync

# 激活虚拟环境 (Windows)
.venv\Scripts\activate

# 激活虚拟环境 (Linux/Mac)
source .venv/bin/activate
```

## Code Quality Tools

项目配置了 Ruff + basedpyright 组合：

### Ruff Commands
```bash
# 检查代码风格和质量
ruff check

# 自动修复可修复的问题
ruff check --fix

# 格式化代码
ruff format

# 检查并格式化
ruff check --fix && ruff format

# 检查特定文件
ruff check model/model_minimind.py
```

### Lint and Format
运行完整的代码质量检查：
```bash
# 完整的代码质量检查流程
ruff check --fix && ruff format
```

## Core Architecture

### Model Structure
- **模型实现**: `model/model_minimind.py` - 包含完整的 MiniMind 模型定义
  - `MiniMindConfig`: 模型配置类，支持标准和 MoE 配置
  - `MiniMindForCausalLM`: 主模型类（因文件截断未完整显示，但基于导入判断存在）
  - `Attention`: 支持 Flash Attention 和 RoPE 位置编码的注意力机制
  - `RMSNorm`: RMS 归一化层

### Training Components
- **训练脚本**: `triainer/train_full_sft.py` - 全参数 SFT 训练
  - 支持分布式训练 (DDP)
  - 支持混合精度训练
  - 集成 WandB 日志记录
  - 梯度累积和裁剪

### Key Features
- **RoPE 位置编码**: 旋转位置编码实现
- **Flash Attention**: 高效注意力计算
- **MoE 支持**: 混合专家模型配置
- **分布式训练**: 支持多 GPU 训练

## Model Configuration

模型配置参数（在 `MiniMindConfig` 中）：
- `hidden_size`: 隐藏层大小 (默认由训练脚本设置)
- `num_hidden_layers`: Transformer 层数 (默认8)
- `num_attention_heads`: 注意力头数 (默认8)
- `num_key_value_heads`: KV 头数，支持 GQA (默认2)
- `max_position_embeddings`: 最大序列长度 (默认32768)
- `vocab_size`: 词汇表大小 (默认6400)
- `use_moe`: 是否使用混合专家模型
- `flash_attn`: 是否使用 Flash Attention

## Training Commands

### Full SFT Training
```bash
# 基本训练
python triainer/train_full_sft.py \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 5e-7 \
    --hidden_size 512 \
    --num_hidden_layers 8

# 分布式训练
torchrun --nproc_per_node=2 triainer/train_full_sft.py \
    --ddp \
    --epochs 2 \
    --batch_size 16

# 使用 WandB 记录
python triainer/train_full_sft.py \
    --use_wandb \
    --wandb_project "MiniMind-Full-SFT"

# MoE 训练
python triainer/train_full_sft.py \
    --use_moe True \
    --hidden_size 512
```

### Training Parameters
- `--data_path`: 训练数据路径 (默认: `../dataset/sft_mini_512.jsonl`)
- `--out_dir`: 模型保存目录 (默认: `../out`)
- `--max_seq_len`: 最大序列长度 (默认: 512)
- `--accumulation_steps`: 梯度累积步数 (默认: 1)
- `--grad_clip`: 梯度裁剪阈值 (默认: 1.0)
- `--save_interval`: 模型保存间隔 (默认: 100步)
- `--log_interval`: 日志输出间隔 (默认: 100步)

## Data Requirements

训练脚本期望数据格式：
- 数据文件：JSONL 格式
- 默认路径：`../dataset/sft_mini_512.jsonl`
- 需要与 `SFTDataset` 类兼容（从 `dataset.lm_dataset` 导入）

## Model Checkpoints

模型检查点保存格式：
- 预训练模型：`{save_dir}/pretrain_{hidden_size}[_moe].pth`
- SFT 模型：`{save_dir}/full_stf_{hidden_size}[_moe].pth`
- 模型以半精度（.half()）格式保存

## Dependencies

主要依赖包括：
- PyTorch >= 2.3.0 (CUDA 12.4 支持)
- Transformers >= 4.44.0
- Accelerate >= 1.0.1
- PEFT >= 0.7.1 (参数高效微调)
- TRL >= 0.13.0 (强化学习训练)
- WandB >= 0.18.3 (实验追踪)

## Development Notes

- 项目结构相对简单，主要包含模型定义和训练脚本
- 训练脚本中存在一些拼写错误（如 `fileterwarning`, `learing_rate` 等）
- 支持 CPU 和 GPU 训练，自动检测 CUDA 可用性
- 使用 cosine 学习率调度策略

## IDE Configuration

项目已配置完整的开发工具链：

### Ruff (代码质量)
- **Linting** - 代码风格检查 (pycodestyle, pyflakes, etc.)
- **Formatting** - 自动格式化代码
- **Import sorting** - 自动排序导入语句
- **性能** - 极快的检查速度

### basedpyright (类型检查)
- **类型推断** - 实时类型分析和检查
- **智能提示** - 基于类型的代码补全
- **错误诊断** - 深度静态分析
- **优化配置** - 减少内存使用，保持对整个项目的分析覆盖

### 推荐工作流
```bash
# 开发前运行代码质量检查
ruff check --fix && ruff format

# IDE 会自动提供类型检查和智能提示
# 保存时可配置自动运行 ruff format
```
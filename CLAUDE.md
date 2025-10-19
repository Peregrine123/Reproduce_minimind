# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

这是一个 MiniMind LLM 的复现项目，包含了一个基于 Transformer 架构的小型语言模型实现。项目支持标准的 Transformer 结构以及混合专家（MoE）模式。

**核心组件**：
- 模型定义：`model/model_minimind.py` (466行)
- 数据处理：`dataset/lm_dataset.py` (233行)
- 训练流程：`triainer/train_pretrian.py` (211行) + `triainer/train_full_sft.py` (230行)
- 统一入口：`main.py` (144行) - 支持 Kaggle/Jupyter 环境

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

## Training Workflow

### 优化后的推荐配置（针对 T4 x 2）

```bash
# 预训练（优化版）
python main.py --mode pretrain \
    --epochs 5 \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --warmup_iters 2000 \
    --accumulation_steps 1

# 有效 batch_size = 32 × 1 × 2 = 64
# 预计训练时间：约 25 小时 (基于实测 2 epoch = 10 小时)
```

**关键改进**：
- ✅ Epochs 从 2 增加到 5（训练更充分，避免欠拟合）
- ✅ Batch size 从 16 增加到 32（充分利用 T4 显存）
- ✅ Learning rate 从 5e-4 降至 2e-4（提高稳定性）
- ✅ Warmup 从 0 增加到 2000 步（避免初期震荡）
- ✅ Accumulation steps 设为 1（参数更新更频繁，训练更快，有效batch=64适合小模型）

📄 **详细分析**：参见 `TRAINING_OPTIMIZATION.md`

### 方法一：使用统一入口 main.py（推荐用于 Kaggle/Jupyter）

```bash
# 预训练（默认模式）
python main.py --mode pretrain \
    --epochs 2 \
    --batch_size 32 \
    --learning_rate 5e-4 \
    --hidden_size 512 \
    --pretrain_data_path ./dataset/pretrain_hq.jsonl

# 监督微调 (SFT)
python main.py --mode sft \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 5e-7 \
    --hidden_size 512 \
    --sft_data_path ./dataset/sft_data.jsonl

# MoE 训练
python main.py --mode pretrain \
    --use_moe True \
    --hidden_size 512

# 分布式训练（需要配合 torchrun）
torchrun --nproc_per_node=2 main.py \
    --mode pretrain \
    --ddp \
    --batch_size 32
```

**main.py 特点**：
- 自动处理 Jupyter/Colab 环境的特殊参数（`-f` 和 `.json` 文件）
- 使用 `exec()` 直接执行训练脚本，解决 `__file__` 未定义问题
- 统一参数接口，简化命令行操作

### 方法二：直接调用训练脚本

```bash
# 预训练
python triainer/train_pretrian.py \
    --data_path ./dataset/pretrain_hq.jsonl \
    --epochs 2 \
    --batch_size 32 \
    --learning_rate 5e-4

# SFT 训练
python triainer/train_full_sft.py \
    --data_path ./dataset/sft_data.jsonl \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 5e-7

# 分布式训练
torchrun --nproc_per_node=2 triainer/train_pretrian.py \
    --ddp \
    --batch_size 32
```

### 通用训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--out_dir` | `./out` | 模型保存目录 |
| `--epochs` | `2` | 训练轮数 |
| `--batch_size` | `16` (SFT) / `32` (Pretrain) | 批次大小 |
| `--learning_rate` | `5e-7` (SFT) / `5e-4` (Pretrain) | 学习率 |
| `--hidden_size` | `512` | 隐藏层维度 |
| `--num_hidden_layers` | `8` | Transformer 层数 |
| `--max_seq_len` | `512` | 最大序列长度 |
| `--use_moe` | `False` | 是否使用 MoE |
| `--accumulation_steps` | `1` | 梯度累积步数 (推荐1，有效batch=32×1×2=64) |
| `--grad_clip` | `1.0` | 梯度裁剪阈值 |
| `--warmup_iters` | `0` | 学习率预热步数 |
| `--save_interval` | `500` | 模型保存间隔（步） |
| `--log_interval` | `100` | 日志输出间隔（步） |
| `--use_wandb` | - | 启用 WandB 日志 |
| `--ddp` | - | 启用分布式训练 |

## Code Quality Tools

项目配置了 Ruff + basedpyright 组合：

```bash
# 检查代码风格和质量
ruff check

# 自动修复可修复的问题
ruff check --fix

# 格式化代码
ruff format

# 完整的代码质量检查流程
ruff check --fix && ruff format

# 检查特定文件
ruff check model/model_minimind.py
```

**Ruff 配置要点**：
- 行长度限制：88 字符
- 启用规则：pycodestyle, pyflakes, isort, pep8-naming, pyupgrade, mccabe, pylint, flake8-bugbear
- Import 排序：自动对 `model` 和 `triainer` 包进行优先排序
- 忽略规则：E501（行长），PLR0913/0912/0915（复杂度），N803/N806（命名）

## Core Architecture

### 模型结构 (`model/model_minimind.py`)

**核心类**：
- `MiniMindConfig`: 模型配置类，支持标准和 MoE 配置
- `MiniMindForCausalLM`: 主模型类，用于因果语言建模
- `MiniMindModel`: Transformer 主体，多层堆叠
- `MiniMindBlock`: 单个 Transformer 块（Attention + FFN）
- `Attention`: 支持 Flash Attention 和 RoPE 位置编码
- `FeedForward`: SwiGLU 前馈网络
- `MoEGate` + `MoEFeedForward`: 混合专家模块
- `RMSNorm`: RMS 归一化层

**关键技术**：
- **RoPE 位置编码**：旋转位置编码（`rope_theta=1e6`）
- **Flash Attention**：高效注意力计算，支持分页 KV 缓存
- **GQA**：分组查询注意力（`num_key_value_heads=2`，`num_attention_heads=8`）
- **MoE**：
  - `num_experts_per_tok=2`：每个 token 选择 2 个专家
  - `n_routed_experts=4`：总共 4 个路由专家
  - `n_shared_experts=1`：1 个共享专家
  - 辅助损失（aux_loss）：平衡专家负载

**模型配置默认值**：
```python
hidden_size=512
num_hidden_layers=8
num_attention_heads=8
num_key_value_heads=2
max_position_embeddings=32768
vocab_size=6400
intermediate_size=1536  # 3 * hidden_size
```

### 数据处理 (`dataset/lm_dataset.py`)

**支持的数据集类型**：
1. `PretrianDataset`：预训练数据集（JSONL 格式）
2. `SFTDataset`：监督微调数据集（ChatML 格式）
3. `DPODataset`：直接偏好优化数据集
4. `RLAIFDataset`：强化学习微调数据集

**数据格式**：
- 输入：JSONL 格式（每行一个 JSON 对象）
- 输出：`(X, Y, loss_mask)` 三元组
  - `X`：输入 token ID
  - `Y`：目标 token ID
  - `loss_mask`：损失掩码（区分真实 token 和 padding）
- 聊天模板：ChatML 格式（`<|im_start|>user/assistant/system<|im_end|>`）

**Tokenizer**：
- 词汇量：6400
- 特殊 token：`<|endoftext|>`(0), `<|im_start|>`(1), `<|im_end|>`(2)
- 文件：`model/tokenizer.json`, `model/vocab.json`, `model/tokenizer_config.json`

### 训练流程架构

**学习率调度**：
- 策略：Cosine Annealing with Warmup
- 实现：`get_lr()` 函数（在 `train_pretrian.py:25` 和 `train_full_sft.py:24`）
- 公式：
  ```python
  # Warmup 阶段
  lr = max_lr * it / warmup_iters

  # Decay 阶段
  decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
  coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
  lr = min_lr + coeff * (max_lr - min_lr)
  ```

**训练循环**：
1. 加载数据 → 2. 前向传播 → 3. 计算损失 → 4. 反向传播 → 5. 梯度累积 → 6. 梯度裁剪 → 7. 优化器更新 → 8. 保存检查点

**混合精度训练**：
- 使用 `torch.cuda.amp.autocast()` 自动混合精度
- 使用 `GradScaler` 防止梯度下溢
- 支持 `bfloat16` 和 `float16`

**分布式训练**：
- 使用 `torch.nn.parallel.DistributedDataParallel` (DDP)
- 自动同步梯度和模型参数
- 支持 `torchrun` 启动

## Model Checkpoints

**保存格式**：
- 预训练模型：`{out_dir}/pretrain_{hidden_size}[_moe].pth`
- SFT 模型：`{out_dir}/full_sft_{hidden_size}[_moe].pth`
- 存储精度：半精度（`.half()`）

**检查点内容**：
- 模型权重：`model.state_dict()`
- 优化器状态：通常不保存（仅推理）

**加载方式**：
```python
# SFT 训练会自动加载对应的预训练模型
pretrain_model_path = f"{out_dir}/pretrain_{hidden_size}{'_moe' if use_moe else ''}.pth"
model.load_state_dict(torch.load(pretrain_model_path))
```

## Data Requirements

**预训练数据**：
- 文件：`dataset/pretrain_hq.jsonl` (1.6GB)
- 格式：每行一个 JSON 对象，包含 `text` 字段
- 示例：`{"text": "这是一段预训练文本..."}`

**SFT 数据**：
- 文件：`dataset/sft_data.jsonl` (用户需自行准备)
- 格式：每行一个 JSON 对象，包含 `conversations` 字段
- 示例：
  ```json
  {
    "conversations": [
      {"role": "user", "content": "你好"},
      {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"}
    ]
  }
  ```

## Dependencies

**核心依赖**：
- PyTorch >= 2.3.0 (CUDA 12.4 支持)
- Transformers >= 4.44.0
- Accelerate >= 1.0.1
- PEFT >= 0.7.1
- TRL >= 0.13.0
- WandB >= 0.18.3

**完整依赖**：参考 `pyproject.toml`，包含数据处理（pandas, jieba, nltk）、可视化（matplotlib, streamlit）、API（flask, openai）等

## Development Notes

**已知问题**：
- 训练脚本所在目录拼写为 `triainer`（应为 `trainer`）
- SFT 模型保存文件名拼写为 `full_stf`（应为 `full_sft`）
- 代码中存在一些拼写错误（如 `fileterwarning`, `learing_rate`）

**重要修复 (2025-10-18)**：
- ✅ **Loss 显示逻辑错误已修复**：旧版本中训练日志显示的 loss 被错误地乘以了 `accumulation_steps`，导致显示值异常高（200+）。实际训练是成功的，真实 loss 在正常范围（7-9）。
- ✅ **验证结果**：预训练模型测试显示 loss 从 8.92 降至 7.42（16.81% 改善），困惑度从 7445 降至 1664。
- ✅ **模型 forward() 方法**：添加了默认参数，使其兼容 transformers 的 generate() 方法。
- 📄 **详细分析**：参见 `LOSS_ANALYSIS.md`

**环境兼容性**：
- 支持 CPU 和 GPU 训练，自动检测 CUDA 可用性
- 自动处理 Jupyter/Colab 环境的特殊参数
- 兼容 Kaggle 环境的 `__file__` 未定义问题

**优化建议**：
- 预训练时使用较大的批次大小（32）和学习率（5e-4）
- SFT 时使用较小的学习率（5e-7）避免灾难性遗忘
- MoE 模式会增加约 4x 参数量和计算量
- 使用梯度累积可以模拟更大的批次大小
- 本项目是运行在云端的 所以不主张在本地跑完整测试, 如果我们需要做实际上的改动 请你在改动完就是上传git 并push
# 复读机问题分析与超参数优化

## 问题诊断

### 当前状态
- ✅ Loss 从 8.92 降至 7.42 (16.81% 改善)
- ✅ 困惑度从 7445 降至 1664 (78% 改善)
- ❌ 生成文本质量差，只会复读输入

### 根本原因

**训练不充分**：虽然 loss 有所下降，但模型还没有学到足够的语言模式和知识。

具体表现：
1. **Epochs 太少**：只训练了 2 个 epoch
2. **没有 Warmup**：warmup_iters = 0，导致训练初期不稳定
3. **可能的过拟合**：学习率可能过高，导致模型过度拟合训练数据的表面模式

## 针对 T4 x 2 的优化方案

### 硬件环境
- GPU: NVIDIA Tesla T4 x 2
- 显存: 16GB per GPU (总计 32GB)
- 带宽: PCIe Gen3 x16

### 推荐配置

#### 预训练参数

```bash
python main.py --mode pretrain \
    --epochs 5 \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --warmup_iters 2000 \
    --accumulation_steps 4 \
    --max_seq_len 512 \
    --hidden_size 512 \
    --num_hidden_layers 8 \
    --use_moe True \
    --grad_clip 1.0 \
    --save_interval 1000 \
    --log_interval 50
```

#### 关键参数说明

| 参数 | 旧值 | 新值 | 理由 |
|------|------|------|------|
| `epochs` | 2 | **5** | 平衡训练效果和时间成本（5 epoch ≈ 25 小时） |
| `batch_size` | 16 | **32** | 用户指定，T4 x 2 可以支持 |
| `learning_rate` | 5e-4 | **2e-4** | 降低学习率，提高训练稳定性 |
| `warmup_iters` | 0 | **2000** | 添加 warmup 阶段，避免训练初期震荡 |
| `accumulation_steps` | 8 | **4** | batch_size 增大后可以减少累积步数 |
| `log_interval` | 100 | **50** | 更频繁的日志输出，便于监控 |
| `save_interval` | 500 | **1000** | 适当增加保存间隔，减少 IO |

### 有效 Batch Size 计算

```
有效 batch_size = batch_size × accumulation_steps × num_gpus
                = 32 × 4 × 2
                = 256
```

这个有效 batch size 对于 512 隐藏层的模型是合理的。

### 学习率调度

当前使用 Cosine Annealing with Warmup：

```python
# Warmup 阶段 (0 -> 2000 steps)
lr = 2e-4 * (step / 2000)

# Decay 阶段 (2000+ steps)
lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(π × decay_ratio))
```

**优势**：
- 前 2000 步逐渐增加学习率，避免训练初期的不稳定
- 后续使用余弦衰减，平滑降低学习率
- 最终学习率约为 2e-5 (max_lr 的 10%)

## 显存使用估算

### 模型参数
- MoE 模式：约 45M 参数
- 标准模式：约 25M 参数

### 训练显存需求（每张 T4）

| 组件 | 显存占用 |
|------|----------|
| 模型参数 | ~180 MB (fp32) / ~90 MB (fp16) |
| 梯度 | ~180 MB |
| 优化器状态 (AdamW) | ~360 MB |
| 激活值 (batch=32, seq=512) | ~8 GB |
| **总计** | **~9 GB** |

✅ T4 16GB 显存充足，可以支持 batch_size=32

## 训练时长估算

**数据集**：1,413,103 样本

**每个 Epoch（基于实测）**：
```
实际测量: 2 epoch = 10 小时
每个 epoch ≈ 5 小时
```

**5 个 Epochs**：
```
总训练时间 ≈ 25 小时
总训练步数 = 22,080 × 5 = 110,400 步
```

**性能优化**：
- 使用 T4 x 2 DDP 模式，自动并行加速
- 使用 bfloat16 混合精度训练
- 梯度累积减少通信开销

## 其他改进建议

### 1. 数据质量检查
```bash
# 检查数据集中是否有重复样本
python -c "
import json
texts = set()
duplicates = 0
with open('./dataset/pretrain_hq.jsonl', 'r') as f:
    for line in f:
        text = json.loads(line)['text']
        if text in texts:
            duplicates += 1
        texts.add(text)
print(f'Duplicate samples: {duplicates}')
"
```

### 2. 生成参数调整

在测试生成时使用以下参数：

```python
output = model.generate(
    input_ids,
    max_new_tokens=50,
    temperature=1.0,          # 提高温度，增加多样性
    top_k=50,
    top_p=0.95,               # 提高 top_p
    repetition_penalty=1.2,   # 添加重复惩罚
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)
```

### 3. 监控指标

训练时关注：
- Loss 下降曲线应该平滑
- 每 1000 步做一次生成测试
- 保存多个检查点，选择最佳的

### 4. Early Stopping

如果 loss 不再下降或生成质量不改善，可以提前停止。

## 预期效果

训练 5 个 epoch 后（约 25 小时）：
- ✅ Loss 应降至 6-7 左右
- ✅ 困惑度应降至 400-1100
- ✅ 生成文本应该有基本的语法和语义连贯性
- ✅ 显著改善"复读机"问题

**当前状态（2 epoch）**：
- Loss: 7.42
- 困惑度: 1664
- 问题: 生成文本质量差，只会复读

**预期改善（5 epoch）**：
- Loss 再降低 0.5-1.0
- 困惑度降至 1000 以下
- 生成文本有基本连贯性

如果 5 epoch 效果仍不理想，考虑：
1. 继续训练到 7-8 epoch（权衡时间成本）
2. 调整数据质量（过滤低质量样本）
3. 调整学习率调度（延长 warmup，调整衰减曲线）
4. 检查数据集是否有大量重复样本

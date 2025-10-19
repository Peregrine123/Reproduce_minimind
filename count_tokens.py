"""统计预训练数据集的token数量"""
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

print("=" * 80)
print("预训练数据集 Token 统计")
print("=" * 80)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("./model/")
print(f"\n1. Tokenizer 信息:")
print(f"   Vocab size: {tokenizer.vocab_size}")
print(f"   Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

# 读取数据集
data_path = "./dataset/pretrain_hq.jsonl"
print(f"\n2. 读取数据集: {data_path}")

total_samples = 0
total_tokens = 0
token_counts = []

print("\n3. 统计中...")
with open(data_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc="Processing samples"):
        sample = json.loads(line)
        text = sample['text']
        tokens = tokenizer(text, add_special_tokens=False)['input_ids']

        total_samples += 1
        token_count = len(tokens)
        total_tokens += token_count
        token_counts.append(token_count)

print(f"\n4. 统计结果:")
print(f"   总样本数: {total_samples:,}")
print(f"   总 Token 数: {total_tokens:,}")
print(f"   平均每个样本 Token 数: {total_tokens / total_samples:.2f}")

print(f"\n5. Token 长度分布:")
token_counts = np.array(token_counts)
print(f"   最小值: {token_counts.min()}")
print(f"   最大值: {token_counts.max()}")
print(f"   中位数: {np.median(token_counts):.0f}")
print(f"   标准差: {token_counts.std():.2f}")

print(f"\n6. 分位数分布:")
percentiles = [10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    value = np.percentile(token_counts, p)
    print(f"   {p}%: {value:.0f} tokens")

print(f"\n7. 长度区间分布:")
bins = [0, 100, 200, 300, 400, 512, 1000, float('inf')]
bin_labels = ['0-100', '101-200', '201-300', '301-400', '401-512', '513-1000', '>1000']
hist, _ = np.histogram(token_counts, bins=bins)
for label, count in zip(bin_labels, hist):
    percentage = count / total_samples * 100
    print(f"   {label:>10} tokens: {count:>8,} ({percentage:>5.2f}%)")

# 计算训练效率
max_seq_len = 512
print(f"\n8. 训练效率分析 (max_seq_len={max_seq_len}):")
truncated = (token_counts > max_seq_len).sum()
print(f"   需要截断的样本数: {truncated:,} ({truncated/total_samples*100:.2f}%)")

# 计算有效token（考虑padding和截断）
effective_tokens = np.minimum(token_counts, max_seq_len)
padding_tokens = (max_seq_len - effective_tokens).sum()
total_capacity = total_samples * max_seq_len
utilization = effective_tokens.sum() / total_capacity * 100

print(f"   有效 token 数: {effective_tokens.sum():,}")
print(f"   Padding token 数: {padding_tokens:,}")
print(f"   总容量 (样本数 × {max_seq_len}): {total_capacity:,}")
print(f"   利用率: {utilization:.2f}%")

# 估算训练规模
print(f"\n9. 训练规模估算:")
epochs = 5
batch_size = 32
num_gpus = 2
accumulation_steps = 4

steps_per_epoch = total_samples // (batch_size * num_gpus)
total_steps = steps_per_epoch * epochs
tokens_per_step = batch_size * num_gpus * max_seq_len
total_training_tokens = total_steps * tokens_per_step

print(f"   Epochs: {epochs}")
print(f"   Batch size: {batch_size} × {num_gpus} GPUs = {batch_size * num_gpus}")
print(f"   Steps per epoch: {steps_per_epoch:,}")
print(f"   Total steps: {total_steps:,}")
print(f"   Tokens per step: {tokens_per_step:,}")
print(f"   Total training tokens: {total_training_tokens:,}")
print(f"   Total training tokens (考虑梯度累积): {total_training_tokens * accumulation_steps:,}")

print("\n" + "=" * 80)
print("统计完成")
print("=" * 80)

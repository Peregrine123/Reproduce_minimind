"""诊断 loss 异常的根本原因"""
import torch
from transformers import AutoTokenizer
from dataset.lm_dataset import PretrianDataset
from model.model_minimind import MiniMindConfig, MiniMindForCasualLM
import torch.nn as nn

print("=" * 80)
print("Loss 异常诊断报告")
print("=" * 80)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("./model/")
print(f"\n1. Tokenizer 信息:")
print(f"   Vocab size: {tokenizer.vocab_size}")
print(f"   Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

# 加载数据集
dataset = PretrianDataset("./dataset/pretrain_hq.jsonl", tokenizer, max_length=512)
print(f"\n2. 数据集信息:")
print(f"   Dataset size: {len(dataset)}")

# 检查一个样本
X, Y, loss_mask = dataset[0]
print(f"\n3. 样本数据:")
print(f"   X shape: {X.shape}")
print(f"   Y shape: {Y.shape}")
print(f"   loss_mask shape: {loss_mask.shape}")
print(f"   有效 tokens: {loss_mask.sum().item()}/{len(loss_mask)}")
print(f"   X 值范围: [{X.min().item()}, {X.max().item()}]")
print(f"   Y 值范围: [{Y.min().item()}, {Y.max().item()}]")

# 检查 Y 中是否有超出 vocab_size 的值
invalid_tokens = (Y >= tokenizer.vocab_size).sum().item()
print(f"   Y 中超出 vocab_size 的 token 数: {invalid_tokens}")
if invalid_tokens > 0:
    print(f"   ⚠️  警告：Y 中有 {invalid_tokens} 个 token ID >= {tokenizer.vocab_size}!")

# 加载模型
config = MiniMindConfig(hidden_size=512, num_hidden_layers=8, use_moe=False)
model = MiniMindForCasualLM(config)
model.eval()

print(f"\n4. 模型配置:")
print(f"   vocab_size: {config.vocab_size}")
print(f"   hidden_size: {config.hidden_size}")
print(f"   num_hidden_layers: {config.num_hidden_layers}")

if config.vocab_size != tokenizer.vocab_size:
    print(f"   ⚠️  警告：模型 vocab_size ({config.vocab_size}) != tokenizer vocab_size ({tokenizer.vocab_size})")

# 测试前向传播
X = X.unsqueeze(0)
Y = Y.unsqueeze(0)
loss_mask = loss_mask.unsqueeze(0)
attention_mask = (X != tokenizer.pad_token_id).long()

print(f"\n5. 前向传播测试:")
with torch.no_grad():
    res = model(
        input_ids=X,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=slice(None)
    )

    print(f"   Logits shape: {res.logits.shape}")
    print(f"   Logits 值范围: [{res.logits.min().item():.3f}, {res.logits.max().item():.3f}]")

    # 按照训练代码计算 loss
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(
        res.logits.view(-1, res.logits.size(-1)),
        Y.view(-1),
    ).view(Y.size())

    print(f"\n6. Loss 计算 (逐 token):")
    print(f"   Loss shape: {loss.shape}")
    print(f"   Loss 值范围: [{loss.min().item():.3f}, {loss.max().item():.3f}]")
    print(f"   Loss 平均值 (所有 token): {loss.mean().item():.3f}")

    # 只计算有效 token
    avg_loss = (loss * loss_mask).sum() / loss_mask.sum()
    print(f"\n7. Loss 计算 (有效 token):")
    print(f"   平均 loss: {avg_loss.item():.3f}")
    print(f"   理论初始值: {torch.tensor(config.vocab_size).log().item():.3f} (ln({config.vocab_size}))")

    # 检查 softmax 概率分布
    probs = torch.softmax(res.logits[0, :5, :], dim=-1)  # 前5个token的概率
    print(f"\n8. Softmax 概率分布 (前5个token):")
    for i in range(5):
        top5_probs, top5_idx = probs[i].topk(5)
        print(f"   Token {i}:")
        print(f"     Top 5 预测: {top5_idx.tolist()}")
        print(f"     Top 5 概率: {[f'{p:.4f}' for p in top5_probs.tolist()]}")
        print(f"     真实 token: {Y[0, i].item()}, 概率: {probs[i, Y[0, i]].item():.6f}")

print("\n" + "=" * 80)
print("诊断完成")
print("=" * 80)

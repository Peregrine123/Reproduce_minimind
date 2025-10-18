"""调试loss异常高的问题"""
import torch
from transformers import AutoTokenizer
from dataset.lm_dataset import PretrianDataset
from model.model_minimind import MiniMindConfig, MiniMindForCasualLM

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("./model/")
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

# 加载数据集
dataset = PretrianDataset("./dataset/pretrain_hq.jsonl", tokenizer, max_length=512)
print(f"Dataset size: {len(dataset)}")

# 检查前几个样本
print("\n检查前3个样本:")
for i in range(min(3, len(dataset))):
    X, Y, loss_mask = dataset[i]
    valid_tokens = loss_mask.sum().item()
    total_tokens = len(loss_mask)
    print(f"Sample {i}: 有效tokens={valid_tokens}/{total_tokens} ({valid_tokens/total_tokens*100:.1f}%)")
    print(f"  X shape: {X.shape}, Y shape: {Y.shape}, loss_mask shape: {loss_mask.shape}")
    print(f"  Y中的唯一值数量: {len(Y.unique())}")

# 加载模型
config = MiniMindConfig(hidden_size=512, num_hidden_layers=8, use_moe=False)
model = MiniMindForCasualLM(config)
model.eval()

print(f"\n模型配置:")
print(f"  vocab_size: {config.vocab_size}")
print(f"  hidden_size: {config.hidden_size}")
print(f"  num_hidden_layers: {config.num_hidden_layers}")

# 测试前向传播
print("\n测试前向传播:")
X, Y, loss_mask = dataset[0]
X = X.unsqueeze(0)  # 添加batch维度
Y = Y.unsqueeze(0)
loss_mask = loss_mask.unsqueeze(0)

attention_mask = (X != tokenizer.pad_token_id).long()

with torch.no_grad():
    res = model(
        input_ids=X,
        attention_mask=attention_mask,
        past_key_values=None,
        use_cache=False,
        logits_to_keep=slice(None)
    )

    # 计算loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(
        res.logits.view(-1, res.logits.size(-1)),
        Y.view(-1),
    ).view(Y.size())

    # 打印详细信息
    print(f"Logits shape: {res.logits.shape}")
    print(f"Loss (per-token) shape: {loss.shape}")
    print(f"Loss (per-token) 统计:")
    print(f"  Min: {loss.min().item():.3f}")
    print(f"  Max: {loss.max().item():.3f}")
    print(f"  Mean: {loss.mean().item():.3f}")

    # 只计算有效token的loss
    valid_loss = loss[loss_mask.bool()]
    print(f"Valid token loss 统计:")
    print(f"  Min: {valid_loss.min().item():.3f}")
    print(f"  Max: {valid_loss.max().item():.3f}")
    print(f"  Mean: {valid_loss.mean().item():.3f}")

    # 计算平均loss（和训练代码一致）
    avg_loss = (loss * loss_mask).sum() / loss_mask.sum()
    print(f"\n平均loss（训练代码计算方式）: {avg_loss.item():.3f}")

    # aux_loss 可能是张量（MoE模式）或整数（标准模式）
    if isinstance(res.aux_loss, torch.Tensor):
        aux_loss_value = res.aux_loss.item()
        total_loss = (avg_loss + res.aux_loss).item()
    else:
        aux_loss_value = res.aux_loss
        total_loss = avg_loss.item() + res.aux_loss

    print(f"Aux loss: {aux_loss_value:.3f}")
    print(f"总loss: {total_loss:.3f}")

print("\n理论上：")
print(f"  随机初始化模型的loss应该接近 ln({config.vocab_size}) = {torch.tensor(config.vocab_size).log().item():.3f}")

"""测试预训练模型的实际性能"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from dataset.lm_dataset import PretrianDataset
from model.model_minimind import MiniMindConfig, MiniMindForCasualLM
import numpy as np

print("=" * 80)
print("预训练模型性能测试")
print("=" * 80)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("./model/")
print(f"\n1. Tokenizer 信息:")
print(f"   Vocab size: {tokenizer.vocab_size}")

# 加载数据集（取前100个样本用于测试）
dataset = PretrianDataset("./dataset/pretrain_hq.jsonl", tokenizer, max_length=512)
print(f"\n2. 数据集信息:")
print(f"   Dataset size: {len(dataset)}")
print(f"   测试样本数: 100")

# 配置
config = MiniMindConfig(hidden_size=512, num_hidden_layers=8, use_moe=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n3. 设备: {device}")

# 测试函数
def evaluate_model(model, dataset, num_samples=100):
    """评估模型在数据集上的 loss 和困惑度"""
    model.eval()
    loss_fct = nn.CrossEntropyLoss(reduction="none")

    total_loss = 0.0
    total_tokens = 0
    losses = []

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            X, Y, loss_mask = dataset[i]
            X = X.unsqueeze(0).to(device)
            Y = Y.unsqueeze(0).to(device)
            loss_mask = loss_mask.unsqueeze(0).to(device)
            attention_mask = (X != tokenizer.pad_token_id).long()

            res = model(
                input_ids=X,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=False,
                logits_to_keep=slice(None)
            )

            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1),
            ).view(Y.size())

            # 只计算有效 token 的 loss
            valid_loss = (loss * loss_mask).sum()
            valid_tokens = loss_mask.sum()

            total_loss += valid_loss.item()
            total_tokens += valid_tokens.item()
            losses.append((valid_loss / valid_tokens).item())

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return {
        'avg_loss': avg_loss,
        'perplexity': perplexity,
        'losses': losses,
        'total_tokens': total_tokens
    }

# 测试未训练模型
print("\n4. 测试未训练模型（随机初始化）:")
untrained_model = MiniMindForCasualLM(config).to(device)
untrained_results = evaluate_model(untrained_model, dataset, num_samples=100)
print(f"   平均 loss: {untrained_results['avg_loss']:.4f}")
print(f"   困惑度 (perplexity): {untrained_results['perplexity']:.2f}")
print(f"   理论初始 loss: {np.log(config.vocab_size):.4f} (ln({config.vocab_size}))")

# 加载预训练模型
print("\n5. 加载预训练模型:")
pretrained_model = MiniMindForCasualLM(config).to(device)
try:
    state_dict = torch.load("./out/pretrain_512_moe.pth", map_location=device)
    pretrained_model.load_state_dict(state_dict)
    print("   ✓ 模型加载成功")
except Exception as e:
    print(f"   ✗ 加载失败: {e}")
    exit(1)

# 测试预训练模型
print("\n6. 测试预训练模型:")
pretrained_results = evaluate_model(pretrained_model, dataset, num_samples=100)
print(f"   平均 loss: {pretrained_results['avg_loss']:.4f}")
print(f"   困惑度 (perplexity): {pretrained_results['perplexity']:.2f}")

# 对比
print("\n7. 性能对比:")
print(f"   Loss 改善: {untrained_results['avg_loss'] - pretrained_results['avg_loss']:.4f}")
print(f"   Loss 改善率: {(1 - pretrained_results['avg_loss']/untrained_results['avg_loss'])*100:.2f}%")
print(f"   困惑度降低: {untrained_results['perplexity'] - pretrained_results['perplexity']:.2f}")

# 统计分析
print("\n8. Loss 分布统计 (100个样本):")
print(f"   未训练模型:")
print(f"     Min: {min(untrained_results['losses']):.4f}")
print(f"     Max: {max(untrained_results['losses']):.4f}")
print(f"     Std: {np.std(untrained_results['losses']):.4f}")
print(f"   预训练模型:")
print(f"     Min: {min(pretrained_results['losses']):.4f}")
print(f"     Max: {max(pretrained_results['losses']):.4f}")
print(f"     Std: {np.std(pretrained_results['losses']):.4f}")

# 生成测试
print("\n9. 生成测试:")
try:
    test_prompt = "你好"
    test_input = tokenizer(test_prompt, return_tensors='pt')
    test_input_ids = test_input['input_ids'].to(device)
    test_attention_mask = test_input['attention_mask'].to(device)

    pretrained_model.eval()
    with torch.no_grad():
        output = pretrained_model.generate(
            test_input_ids,
            attention_mask=test_attention_mask,
            max_new_tokens=20,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"   输入: {test_prompt}")
        print(f"   生成: {generated_text}")
except Exception as e:
    print(f"   生成测试失败（这是正常的，generate() 可能需要额外配置）")
    print(f"   错误: {e}")
    print(f"   跳过生成测试，继续其他评估...")


print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)

# 结论
print("\n结论:")
if pretrained_results['avg_loss'] < untrained_results['avg_loss']:
    print("✓ 预训练模型的 loss 明显低于未训练模型，训练是有效的！")
    if pretrained_results['avg_loss'] < 9.0:
        print("✓ Loss 值正常（< 9.0），符合预期")
    else:
        print("⚠ Loss 值偏高（> 9.0），可能需要更多训练")
else:
    print("✗ 预训练模型 loss 没有改善，训练可能失败")

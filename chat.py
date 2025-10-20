"""
交互式对话脚本 - 用于测试 MiniMind 模型权重

使用方法：
    python chat.py --model_path out/pretrain_512_moe.pth
    python chat.py --model_path out/full_sft_512.pth --hidden_size 512
"""

import argparse
import os
import sys
import torch
from transformers import AutoTokenizer

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def load_model(model_path, hidden_size=512, use_moe=None, device='cuda'):
    """加载模型和权重"""

    # 自动检测是否为 MoE 模型
    if use_moe is None:
        use_moe = '_moe' in model_path

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 错误：模型文件不存在: {model_path}")
        print("\n可用的模型文件：")
        out_dir = os.path.dirname(model_path) or './out'
        if os.path.exists(out_dir):
            for f in os.listdir(out_dir):
                if f.endswith('.pth'):
                    print(f"  - {os.path.join(out_dir, f)}")
        sys.exit(1)

    # 创建模型配置
    # intermediate_size 使用与训练时相同的计算逻辑
    # intermediate_size = int(hidden_size * 8 / 3) 然后向上取整到64的倍数
    intermediate_size_raw = int(hidden_size * 8 / 3)
    intermediate_size = 64 * ((intermediate_size_raw + 64 - 1) // 64)

    config = MiniMindConfig(
        hidden_size=hidden_size,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=intermediate_size,
        vocab_size=6400,
        use_moe=use_moe,
    )

    print(f"📦 加载模型配置：")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Intermediate size: {intermediate_size}")
    print(f"  - Layers: {config.num_hidden_layers}")
    print(f"  - MoE: {'✅ 启用' if use_moe else '❌ 禁用'}")
    print(f"  - Vocab size: {config.vocab_size}")

    # 创建模型
    model = MiniMindForCausalLM(config)

    # 加载权重
    print(f"\n🔄 加载模型权重: {model_path}")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # 移动到设备并设置为评估模式
    model = model.to(device)
    model.eval()

    print(f"✅ 模型加载成功！设备: {device}")

    return model, config


def load_tokenizer(tokenizer_path='./model'):
    """加载 tokenizer"""
    print(f"\n🔤 加载 Tokenizer: {tokenizer_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=True
        )
        print(f"✅ Tokenizer 加载成功！词汇量: {len(tokenizer)}")
        return tokenizer
    except Exception as e:
        print(f"❌ 加载 Tokenizer 失败: {e}")
        sys.exit(1)


def format_chat_prompt(messages, tokenizer):
    """将消息列表格式化为 ChatML 格式"""
    # 使用 tokenizer 的 chat_template 进行格式化
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt


def generate_response(model, tokenizer, messages, max_new_tokens=256, temperature=0.7, top_p=0.9, top_k=50, device='cuda', debug=False):
    """生成回复 - 手动实现生成逻辑以避免兼容性问题"""

    # 格式化输入
    prompt = format_chat_prompt(messages, tokenizer)

    if debug:
        print(f"\n[DEBUG] Prompt: {prompt[:200]}...")
        print(f"[DEBUG] Vocab size: {len(tokenizer)}")
        print(f"[DEBUG] EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        print(f"[DEBUG] BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)

    if debug:
        print(f"[DEBUG] Input IDs shape: {input_ids.shape}")
        print(f"[DEBUG] Input IDs range: [{input_ids.min().item()}, {input_ids.max().item()}]")

    # 手动生成
    generated_tokens = []
    past_key_values = None

    model.eval()
    with torch.no_grad():
        for step in range(max_new_tokens):
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )

            # 获取最后一个位置的 logits
            next_token_logits = outputs.logits[:, -1, :]

            # 温度采样
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

                # Top-k 过滤
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')

                # Top-p (nucleus) 过滤
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                    # 移除累积概率超过 top_p 的 token
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                # 采样
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            next_token_id = next_token.item()

            if debug and step < 10:  # 只显示前10个 token
                token_str = tokenizer.decode([next_token_id], skip_special_tokens=False)
                print(f"[DEBUG] Step {step}: Token ID {next_token_id} -> '{token_str}'")

            # 检查 token ID 是否在有效范围内
            if next_token_id >= len(tokenizer):
                print(f"\n⚠️  警告：生成的 token ID {next_token_id} 超出词汇表范围 [0, {len(tokenizer)-1}]")
                break

            # 检查是否生成了结束 token
            if next_token_id == tokenizer.eos_token_id:
                if debug:
                    print(f"[DEBUG] EOS token generated at step {step}")
                break

            # 添加到生成的 token 列表
            generated_tokens.append(next_token_id)

            # 准备下一轮输入
            input_ids = next_token
            past_key_values = outputs.past_key_values

    # 解码生成的 token
    if generated_tokens:
        if debug:
            print(f"\n[DEBUG] Generated {len(generated_tokens)} tokens")
            print(f"[DEBUG] Token IDs: {generated_tokens[:20]}...")

        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()
    else:
        return ""


def interactive_chat(model, tokenizer, device='cuda', max_new_tokens=256, temperature=0.7, top_p=0.9, top_k=50, debug=False):
    """交互式对话循环"""

    print("\n" + "="*60)
    print("🤖 MiniMind 交互式对话")
    print("="*60)
    print("\n命令：")
    print("  - 输入消息开始对话")
    print("  - 输入 'clear' 清空对话历史")
    print("  - 输入 'debug' 切换调试模式")
    print("  - 输入 'exit' 或 'quit' 退出")
    print("  - 输入 'params' 查看/修改生成参数")
    print("\n生成参数：")
    print(f"  - max_new_tokens: {max_new_tokens}")
    print(f"  - temperature: {temperature}")
    print(f"  - top_p: {top_p}")
    print(f"  - top_k: {top_k}")
    print(f"  - debug: {'✅ 启用' if debug else '❌ 禁用'}")
    print("="*60 + "\n")

    # 对话历史
    messages = []

    while True:
        try:
            # 获取用户输入
            user_input = input("👤 你: ").strip()

            if not user_input:
                continue

            # 处理命令
            if user_input.lower() in ['exit', 'quit']:
                print("\n👋 再见！")
                break

            elif user_input.lower() == 'clear':
                messages = []
                print("🗑️  对话历史已清空\n")
                continue

            elif user_input.lower() == 'debug':
                debug = not debug
                print(f"🐛 调试模式已{'启用' if debug else '禁用'}\n")
                continue

            elif user_input.lower() == 'params':
                print("\n当前生成参数：")
                print(f"  max_new_tokens: {max_new_tokens}")
                print(f"  temperature: {temperature}")
                print(f"  top_p: {top_p}")
                print(f"  top_k: {top_k}")
                print(f"  debug: {'✅ 启用' if debug else '❌ 禁用'}")

                modify = input("\n是否修改参数？(y/n): ").strip().lower()
                if modify == 'y':
                    try:
                        max_new_tokens = int(input(f"max_new_tokens [{max_new_tokens}]: ") or max_new_tokens)
                        temperature = float(input(f"temperature [{temperature}]: ") or temperature)
                        top_p = float(input(f"top_p [{top_p}]: ") or top_p)
                        top_k = int(input(f"top_k [{top_k}]: ") or top_k)
                        print("✅ 参数已更新\n")
                    except ValueError:
                        print("❌ 参数格式错误\n")
                continue

            # 添加用户消息
            messages.append({"role": "user", "content": user_input})

            # 生成回复
            print("🤖 MiniMind: ", end="", flush=True)
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                device=device,
                debug=debug
            )
            print(response + "\n")

            # 添加助手消息
            messages.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 生成失败: {e}\n")
            import traceback
            if debug:
                traceback.print_exc()
            # 移除失败的消息
            if messages and messages[-1]["role"] == "user":
                messages.pop()


def main():
    parser = argparse.ArgumentParser(description='MiniMind 交互式对话脚本')

    # 模型参数
    parser.add_argument('--model_path', type=str, default='out/pretrain_512_moe.pth',
                        help='模型权重路径')
    parser.add_argument('--tokenizer_path', type=str, default='./model',
                        help='Tokenizer 路径')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='隐藏层大小')
    parser.add_argument('--use_moe', type=bool, default=None,
                        help='是否使用 MoE（默认从文件名自动检测）')

    # 生成参数
    parser.add_argument('--max_new_tokens', type=int, default=256,
                        help='最大生成 token 数')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='温度参数（越高越随机）')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='nucleus sampling 参数')
    parser.add_argument('--top_k', type=int, default=50,
                        help='top-k sampling 参数')

    # 设备参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='运行设备')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式，显示详细的生成信息')

    args = parser.parse_args()

    # 显示设备信息
    print(f"\n{'='*60}")
    print("🚀 MiniMind 交互式对话启动")
    print(f"{'='*60}")
    print(f"设备: {args.device}")
    if args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"{'='*60}\n")

    # 加载 tokenizer
    tokenizer = load_tokenizer(args.tokenizer_path)

    # 加载模型
    model, config = load_model(
        model_path=args.model_path,
        hidden_size=args.hidden_size,
        use_moe=args.use_moe,
        device=args.device
    )

    # 开始交互式对话
    interactive_chat(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        debug=args.debug
    )


if __name__ == '__main__':
    main()

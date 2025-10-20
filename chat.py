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


def generate_response(model, tokenizer, messages, max_new_tokens=256, temperature=0.7, top_p=0.9, top_k=50, device='cuda'):
    """生成回复"""

    # 格式化输入
    prompt = format_chat_prompt(messages, tokenizer)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    # 解码（只取生成的部分）
    generated_ids = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return response.strip()


def interactive_chat(model, tokenizer, device='cuda', max_new_tokens=256, temperature=0.7, top_p=0.9, top_k=50):
    """交互式对话循环"""

    print("\n" + "="*60)
    print("🤖 MiniMind 交互式对话")
    print("="*60)
    print("\n命令：")
    print("  - 输入消息开始对话")
    print("  - 输入 'clear' 清空对话历史")
    print("  - 输入 'exit' 或 'quit' 退出")
    print("  - 输入 'params' 查看/修改生成参数")
    print("\n生成参数：")
    print(f"  - max_new_tokens: {max_new_tokens}")
    print(f"  - temperature: {temperature}")
    print(f"  - top_p: {top_p}")
    print(f"  - top_k: {top_k}")
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

            elif user_input.lower() == 'params':
                print("\n当前生成参数：")
                print(f"  max_new_tokens: {max_new_tokens}")
                print(f"  temperature: {temperature}")
                print(f"  top_p: {top_p}")
                print(f"  top_k: {top_k}")

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
                device=device
            )
            print(response + "\n")

            # 添加助手消息
            messages.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 生成失败: {e}\n")
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
        top_k=args.top_k
    )


if __name__ == '__main__':
    main()

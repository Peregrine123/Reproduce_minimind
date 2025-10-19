#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniMind项目主入口文件
提供预训练和SFT训练的统一入口
针对 Kaggle/Jupyter 环境优化
支持自动检测多GPU并启用分布式训练
"""

import argparse
import os
import socket
import subprocess
import sys

import torch


def find_free_port():
    """查找一个可用的端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def main():
    parser = argparse.ArgumentParser(description="MiniMind 训练脚本")
    parser.add_argument(
        "--mode",
        choices=["pretrain", "sft"],
        default="pretrain",
        help="训练模式: pretrain(预训练) 或 sft(监督微调)，默认为pretrain"
    )

    # 通用参数
    parser.add_argument("--out_dir", type=str, default="./out", help="输出目录")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数 (推荐预训练5轮，SFT 2-3轮)")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小 (T4 x2 推荐32)")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率 (预训练2e-4, SFT 5e-5)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用 wandb 记录训练过程")
    parser.add_argument("--wandb_project", type=str, default="MiniMind", help="wandb 项目名称")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载器工作进程数")
    parser.add_argument("--ddp", action="store_true", help="是否启用分布式训练")
    parser.add_argument("--auto_ddp", action="store_true", default=True, help="自动检测多GPU并启用分布式训练")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数 (有效batch=32×1×2=64)")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=2000, help="学习率预热步数 (推荐2000)")
    parser.add_argument("--log_interval", type=int, default=50, help="日志打印间隔 (更频繁的监控)")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument("--local_rank", type=int, default=-1, help="本地进程排名")
    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层层数")
    parser.add_argument("--max_seq_len", default=512, type=int, help="最大序列长度")
    parser.add_argument("--use_moe", default=False, type=bool, help="是否使用混合专家模型")

    # 预训练特定参数
    parser.add_argument("--pretrain_data_path", type=str, default="./dataset/pretrain_hq.jsonl", help="预训练数据路径")

    # SFT特定参数
    parser.add_argument("--sft_data_path", type=str, default="./dataset/sft_data.jsonl", help="SFT训练数据路径")

    # 断点重续参数
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从checkpoint恢复训练的路径")

    # 解析参数，忽略Jupyter/Colab环境自动添加的参数
    args, unknown_args = parser.parse_known_args()

    # 如果有未知参数且是Jupyter/Colab相关的，忽略它们
    # 通常Jupyter/Colab会传递-f参数和JSON文件路径
    if unknown_args and any(arg.startswith('-f') or arg.endswith('.json') for arg in unknown_args):
        # 忽略这些参数，不进行任何处理
        pass

    # 检测是否在 Jupyter/IPython 环境中
    def is_jupyter_environment():
        """检测是否在 Jupyter/IPython 环境中"""
        try:
            # 检查是否有 get_ipython 函数
            get_ipython()
            return True
        except NameError:
            return False
        except:
            # 检查 __file__ 是否存在
            return '__file__' not in globals()

    in_jupyter = is_jupyter_environment()

    # 自动检测多GPU并启用分布式训练
    num_gpus = torch.cuda.device_count()
    if args.auto_ddp and num_gpus > 1 and not args.ddp:
        if in_jupyter:
            # 在 Jupyter 环境中，不能使用 torchrun 自动启动多进程
            print(f"检测到 {num_gpus} 个 GPU，但当前运行在 Jupyter 环境中")
            print(f"Jupyter 环境不支持自动多 GPU 分布式训练")
            print(f"将使用单 GPU 训练模式：{args.device}")
            print(f"\n如需使用多 GPU，请在命令行环境中运行：")
            free_port = find_free_port()
            print(f"  torchrun --nproc_per_node={num_gpus} --master_port={free_port} main.py --mode={args.mode}")
        else:
            print(f"检测到 {num_gpus} 个 GPU，自动启用分布式训练")
            print(f"使用 torchrun 启动分布式训练...")

            # 查找可用端口
            master_port = find_free_port()
            print(f"使用端口: {master_port}")

            # 构建 torchrun 命令
            torchrun_cmd = [
                sys.executable, "-m", "torch.distributed.run",
                f"--nproc_per_node={num_gpus}",
                f"--master_port={master_port}",
                __file__,
                f"--mode={args.mode}",
                f"--out_dir={args.out_dir}",
                f"--epochs={args.epochs}",
                f"--batch_size={args.batch_size}",
                f"--learning_rate={args.learning_rate}",
                f"--device={args.device}",
                f"--dtype={args.dtype}",
                f"--num_workers={args.num_workers}",
                "--ddp",  # 显式启用 DDP
                "--no-auto_ddp",  # 防止递归调用
                f"--accumulation_steps={args.accumulation_steps}",
                f"--grad_clip={args.grad_clip}",
                f"--warmup_iters={args.warmup_iters}",
                f"--log_interval={args.log_interval}",
                f"--save_interval={args.save_interval}",
                f"--local_rank={args.local_rank}",
                f"--hidden_size={args.hidden_size}",
                f"--num_hidden_layers={args.num_hidden_layers}",
                f"--max_seq_len={args.max_seq_len}",
                f"--use_moe={args.use_moe}",
            ]

            if args.mode == "pretrain":
                torchrun_cmd.append(f"--pretrain_data_path={args.pretrain_data_path}")
            else:
                torchrun_cmd.append(f"--sft_data_path={args.sft_data_path}")

            if args.resume_from_checkpoint:
                torchrun_cmd.append(f"--resume_from_checkpoint={args.resume_from_checkpoint}")

            if args.use_wandb:
                torchrun_cmd.append("--use_wandb")
                torchrun_cmd.append(f"--wandb_project={args.wandb_project}")

            # 执行 torchrun 命令
            result = subprocess.run(torchrun_cmd)
            sys.exit(result.returncode)

    # 只在主进程打印（DDP 模式下 rank 0，或非 DDP 模式）
    is_main_process = int(os.environ.get("RANK", 0)) == 0

    if is_main_process:
        if num_gpus > 1 and args.ddp:
            print(f"检测到 {num_gpus} 个 GPU，使用分布式训练模式")
        elif num_gpus == 1:
            print(f"检测到 1 个 GPU，使用单 GPU 训练模式：{args.device}")
        else:
            print(f"未检测到 GPU，使用 CPU 训练模式")

    if args.mode == "pretrain":
        # 构建预训练参数列表
        pretrain_args = [
            "train_pretrian.py",
            f"--out_dir={args.out_dir}",
            f"--epochs={args.epochs}",
            f"--batch_size={args.batch_size}",
            f"--learning_rate={args.learning_rate}",
            f"--device={args.device}",
            f"--dtype={args.dtype}",
            f"--num_workers={args.num_workers}",
            f"--accumulation_steps={args.accumulation_steps}",
            f"--grad_clip={args.grad_clip}",
            f"--warmup_iters={args.warmup_iters}",
            f"--log_interval={args.log_interval}",
            f"--save_interval={args.save_interval}",
            f"--local_rank={args.local_rank}",
            f"--hidden_size={args.hidden_size}",
            f"--num_hidden_layers={args.num_hidden_layers}",
            f"--max_seq_len={args.max_seq_len}",
            f"--use_moe={args.use_moe}",
            f"--data_path={args.pretrain_data_path}"
        ]

        if args.resume_from_checkpoint:
            pretrain_args.append(f"--resume_from_checkpoint={args.resume_from_checkpoint}")

        if args.use_wandb:
            pretrain_args.append("--use_wandb")
            pretrain_args.append(f"--wandb_project={args.wandb_project}-Pretrain")

        if args.ddp:
            pretrain_args.append("--ddp")

        # 修改sys.argv以传递参数并直接执行训练脚本
        sys.argv = pretrain_args

        # 使用 exec 直接执行训练脚本
        script_path = 'triainer/train_pretrian.py'
        with open(script_path, 'r', encoding='utf-8') as f:
            code = f.read()
            # 创建执行环境，包含 __file__ 变量
            exec_globals = {
                '__file__': os.path.abspath(script_path),
                '__name__': '__main__',
            }
            # 移除 if __name__ == "__main__" 检查，直接执行
            exec(code.replace('if __name__ == "__main__":', 'if True:'), exec_globals)

    elif args.mode == "sft":
        # 构建SFT参数列表
        sft_args = [
            "train_full_sft.py",
            f"--out_dir={args.out_dir}",
            f"--epochs={args.epochs}",
            f"--batch_size={args.batch_size}",
            f"--learning_rate={args.learning_rate}",
            f"--device={args.device}",
            f"--dtype={args.dtype}",
            f"--num_workers={args.num_workers}",
            f"--accumulation_steps={args.accumulation_steps}",
            f"--grad_clip={args.grad_clip}",
            f"--warmup_iters={args.warmup_iters}",
            f"--log_interval={args.log_interval}",
            f"--save_interval={args.save_interval}",
            f"--local_rank={args.local_rank}",
            f"--hidden_size={args.hidden_size}",
            f"--num_hidden_layers={args.num_hidden_layers}",
            f"--max_seq_len={args.max_seq_len}",
            f"--use_moe={args.use_moe}",
            f"--data_path={args.sft_data_path}"
        ]

        if args.use_wandb:
            sft_args.append("--use_wandb")
            sft_args.append(f"--wandb_project={args.wandb_project}-SFT")

        if args.ddp:
            sft_args.append("--ddp")

        # 修改sys.argv以传递参数
        sys.argv = sft_args
        from triainer.train_full_sft import main as sft_main
        sft_main()


if __name__ == "__main__":
    main()

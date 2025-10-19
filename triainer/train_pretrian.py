import argparse
import math
import os
import sys
import time
import warnings
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from dataset.lm_dataset import PretrianDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from utils.reproducibility import set_seed


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


warnings.filterwarnings("ignore")


def logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb, start_step=0):
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    start_time = time.time()

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 如果正在恢复训练，跳过已经训练过的步骤
        if step < start_step:
            continue

        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        with ctx:
            # 创建 attention_mask: 1表示有效token，0表示padding
            # 注意：loss_mask 是 Y 的mask，而 X 可能在末尾有padding
            # 所以我们需要从 X 本身计算 attention_mask
            attention_mask = (X != tokenizer.pad_token_id).long()

            # 调用模型，传递所有必需参数
            res = model(
                input_ids=X,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=False,
                logits_to_keep=slice(None)  # 使用所有logits
            )
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1),
            ).view(Y.size())

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss

            # 保存用于显示的实际 loss（在归一化之前）
            actual_loss = loss.item()

            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            logger(
                "Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:".format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    actual_loss,  # 使用真实的 loss 值
                    optimizer.param_groups[-1]["lr"],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                )
            )

            # 如果启用了 wandb 且是主进程，记录训练指标
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log(
                    {
                        "loss": actual_loss,  # 使用真实的 loss 值
                        "lr": optimizer.param_groups[-1]["lr"],
                        "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                    }
                )
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()  # 切换到评估模式

            # 根据是否使用 MoE 设置文件名后缀
            moe_path = "_moe" if lm_config.use_moe else ""

            # 保存最终模型（兼容旧版本）
            ckp = f"{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth"

            # 保存完整的checkpoint（用于断点重续）
            ckp_full = f"{args.save_dir}/checkpoint_epoch{epoch}_step{step + 1}.pth"

            # 获取模型状态字典（处理分布式训练的情况）
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # 保存最终模型（半精度，仅模型权重）
            state_dict_half = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict_half, ckp)

            # 保存完整checkpoint（全精度，包含训练状态）
            checkpoint = {
                'epoch': epoch,
                'step': step + 1,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'lm_config': lm_config,
                'args': vars(args)
            }
            torch.save(checkpoint, ckp_full)
            logger(f"Checkpoint saved: {ckp_full}")

            model.train()  # 切回训练模式

def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained("./model/")
    model = MiniMindForCausalLM(lm_config).to(args.device)
    logger(f"LLM可训练参数量:{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} M")
    return model, tokenizer


def load_checkpoint(checkpoint_path, model, optimizer, scaler):
    """加载checkpoint以恢复训练"""
    logger(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)

    # 加载模型状态
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    # 加载优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 加载scaler状态
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # 恢复位置
    start_epoch = checkpoint['epoch']
    start_step = checkpoint['step']

    # 清空梯度，避免残留值影响重放的小批次
    optimizer.zero_grad(set_to_none=True)

    accumulation_steps = max(1, getattr(args, "accumulation_steps", 1))
    remainder = start_step % accumulation_steps
    if remainder != 0:
        adjusted_step = max(0, start_step - remainder)
        logger(
            f"Checkpoint 在 optimizer.step 前保存：从 step {start_step} 回退到 {adjusted_step}，"
            f"重放 {remainder} 个未完成的 microbatch（accumulation_steps={accumulation_steps}）。"
        )
        start_step = adjusted_step

    logger(f"Checkpoint loaded: resuming from epoch {start_epoch + 1}, step {start_step}")
    return start_epoch, start_step



def init_distributed_mode():
    """初始化分布式训练环境"""
    if not ddp:
        return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")
    parser.add_argument("--out_dir", type=str, default="../out", help="输出目录")
    # 训练轮数：快速测试设为1轮，正式训练建议2-6轮
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用 wandb 记录训练过程")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb 项目名称")
    parser.add_argument("--num_workers", type=int, default=1, help="数据加载器工作进程数")
    parser.add_argument("--ddp", action="store_true", help="是否启用分布式训练")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=0, help="学习率预热步数")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--local_rank", type=int, default=-1, help="本地进程排名")
    parser.add_argument("--hidden_size", default=512, type=int, help="隐藏层维度")
    parser.add_argument("--num_hidden_layers", default=8, type=int, help="隐藏层层数")
    parser.add_argument("--max_seq_len", default=512, type=int, help="最大序列长度")
    parser.add_argument("--use_moe", action="store_true", default=True, help="是否使用混合专家模型（默认启用）")
    parser.add_argument("--data_path", type=str, default="/kaggle/working/dir/pretrain_hq.jsonl", help="训练数据路径")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="从checkpoint恢复训练的路径")
    args = parser.parse_args()

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe
    )

    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = (
        f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
    )

    ctx = nullcontext() if device_type == 'cpu' else torch.cuda.amp.autocast()

    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"

    base_seed = 1337

    # 如果启用 DDP，初始化分布式训练
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        # 设置种子，DDP 模式下每个进程使用不同的种子
        set_seed(base_seed, rank=rank, offset_by_rank=True)
    else:
        # 单机训练，设置统一种子
        set_seed(base_seed)

    if args.use_wandb and (not ddp or dist.get_rank() == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    else:
        wandb = None


    model, tokenizer = init_model(lm_config)

    train_ds = PretrianDataset(args.data_path, tokenizer, max_length=args.max_seq_len)

    # 使用 DistributedSampler 进行分布式数据采样
    train_sampler = DistributedSampler(train_ds) if ddp else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False if ddp else True,  # DDP 时不使用 shuffle，由 sampler 处理
        num_workers=args.num_workers,
        sampler=train_sampler,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ["float16", "bfloat16"]))

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 使用 DistributedDataParallel 包装模型
    if ddp:
        # RoPE 的频率缓冲区不需要在 DDP 中同步（每个进程都是相同的）
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)

    # 初始化起始epoch和step
    start_epoch = 0
    start_step = 0

    # 如果指定了checkpoint路径，则加载checkpoint
    if args.resume_from_checkpoint:
        if os.path.exists(args.resume_from_checkpoint):
            start_epoch, start_step = load_checkpoint(
                args.resume_from_checkpoint, model, optimizer, scaler
            )
        else:
            logger(f"Warning: Checkpoint file not found: {args.resume_from_checkpoint}")
            logger("Starting training from scratch...")

    for epoch in range(start_epoch, args.epochs):
        # 在分布式训练中设置sampler的epoch
        if ddp:
            train_sampler.set_epoch(epoch)

        # 如果是从checkpoint恢复的第一个epoch，传入start_step
        # 否则从step 0开始
        current_start_step = start_step if epoch == start_epoch else 0
        train_epoch(epoch, wandb, start_step=current_start_step)


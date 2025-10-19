import argparse
import math
import os
import sys
import time
import warnings
from contextlib import nullcontext

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.distributed as dist
from dotenv import load_dotenv
from dataset.lm_dataset import SFTDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from utils.reproducibility import set_seed
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

# 加载 .env 文件中的环境变量
load_dotenv()

warnings.filterwarnings("ignore")


def get_lr(current_step, total_step, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_step))


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(
            epoch * iter_per_epoch + step,
            args.epochs * iter_per_epoch,
            args.learning_rate,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        with ctx:
            res = model(X)
            loss = loss_fct(res.logits.view(-1, res.logits.size(-1)), Y.view(-1)).view(
                Y.size()
            )

            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss

            # 保存用于显示的实际 loss（在归一化之前）
            actual_loss = loss.item()

            loss = loss / args.accumulation_steps
        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
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

            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log(
                    {
                        "loss": actual_loss,  # 使用真实的 loss 值
                        "lr": optimizer.param_groups[-1]["lr"],
                        "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60
                        - spend_time // 60,
                    }
                )
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            if args.save_total_limit == 0:
                Logger("跳过权重保存（save_total_limit=0）。")
            else:
                model.eval()
                moe_path = "_moe" if lm_config.use_moe else ""
                ckp = f"{args.save_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth"
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                state_dict = {k: v.half() for k, v in state_dict.items()}
                torch.save(state_dict, ckp)
                model.train()


def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained("../model")
    model = MiniMindForCausalLM(lm_config)
    moe_path = "_moe" if lm_config.use_moe else ""
    ckp = f"{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth"
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)

    Logger(
        f"LLM 可训练参数为: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万"
    )
    model = model.to(args.device)
    return model, tokenizer


def init_distributed_mode():
    if not ddp:
        return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


def main():
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument(
        "--use_wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否使用 wandb 记录训练过程（默认开启，可用 --no-use_wandb 关闭）"
    )
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=5, help="最多保留的checkpoint数量（SFT默认覆盖保存）")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--hidden_size", default=512, type=int)
    parser.add_argument("--num_hidden_layers", default=8, type=int)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--use_moe", action="store_true", default=True, help="是否使用混合专家模型（默认启用）")
    parser.add_argument(
        "--data_path", type=str, default="../dataset/sft_mini_512.jsonl"
    )

    global args
    args = parser.parse_args()

    global lm_config
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe,
    )

    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSzie-{args.batch_size}-LearningRate-{args.learning_rate}"
    global ctx
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    global ddp
    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"
    base_seed = 42

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        # 设置种子，DDP 模式下每个进程使用不同的种子
        set_seed(base_seed, rank=rank, offset_by_rank=True)
    else:
        # 单机训练，设置统一种子
        set_seed(base_seed)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    global model, tokenizer
    model, tokenizer = init_model(lm_config)

    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    global train_loader
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler,
    )

    global optimizer, iter_per_epoch
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ["float16", "bfloat16"]))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    if ddp:
        # RoPE 的频率缓冲区不需要在 DDP 中同步（每个进程都是相同的）
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)


if __name__ == "__main__":
    main()

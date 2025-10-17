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
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.lm_dataset import PretrianDataset
from model.model_minimind import MiniMindConfig, MiniMindForCasualLM


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


warnings.filterwarnings("ignore")


def logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    start_time = time.time()

    for step, (X, Y, loss_mask) in enumerate(train_loader):
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
                    loss.item() * args.accumulation_steps,  # 恢复实际损失值
                    optimizer.param_groups[-1]["lr"],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                )
            )

            # 如果启用了 wandb 且是主进程，记录训练指标
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log(
                    {
                        "loss": loss.item() * args.accumulation_steps,
                        "lr": optimizer.param_groups[-1]["lr"],
                        "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                    }
                )
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()  # 切换到评估模式

            # 根据是否使用 MoE 设置文件名后缀
            moe_path = "_moe" if lm_config.use_moe else ""
            ckp = f"{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth"

            # 获取模型状态字典（处理分布式训练的情况）
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            # 转换为半精度以节省存储空间
            state_dict = {k: v.half() for k, v in state_dict.items()}
            torch.save(state_dict, ckp)
            model.train()  # 切回训练模式

def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained("./model/")
    model = MiniMindForCasualLM(lm_config).to(args.device)
    logger(f"LLM可训练参数量:{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} M")
    return model, tokenizer


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
    parser.add_argument("--use_moe", default=False, type=bool, help="是否使用混合专家模型")
    parser.add_argument("--data_path", type=str, default="/kaggle/working/dir/pretrain_hq.jsonl", help="训练数据路径")
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
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if args.use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    else:
        wandb = None


    model, tokenizer = init_model(lm_config)

    train_ds = PretrianDataset(args.data_path, tokenizer, max_length=args.max_seq_len)


    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ["float16", "bfloat16"]))

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    iter_per_epoch = len(train_loader)

    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)

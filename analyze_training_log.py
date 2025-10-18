"""分析训练日志中的loss变化"""
import re

# 读取日志文件
with open("下载.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# 提取loss数据
losses = []
steps = []
pattern = r"Epoch:\[(\d+)/\d+\]\((\d+)/\d+\) loss:([\d.]+)"

for line in lines:
    match = re.search(pattern, line)
    if match:
        epoch = int(match.group(1))
        step = int(match.group(2))
        loss = float(match.group(3))
        steps.append(step)
        losses.append(loss)

print(f"总共找到 {len(losses)} 个 loss 记录")
print(f"\nLoss 统计:")
print(f"  最小值: {min(losses):.3f}")
print(f"  最大值: {max(losses):.3f}")
print(f"  平均值: {sum(losses)/len(losses):.3f}")
print(f"  开始时: {losses[0]:.3f}")
print(f"  结束时: {losses[-1]:.3f}")

# 统计异常高的loss
high_loss_count = sum(1 for loss in losses if loss > 200)
print(f"\n异常高的 loss (>200): {high_loss_count}/{len(losses)} ({high_loss_count/len(losses)*100:.1f}%)")

very_high_loss_count = sum(1 for loss in losses if loss > 250)
print(f"非常高的 loss (>250): {very_high_loss_count}/{len(losses)} ({very_high_loss_count/len(losses)*100:.1f}%)")

# 检查前10个loss
print(f"\n前10个 loss:")
for i in range(min(10, len(losses))):
    print(f"  Step {steps[i]}: {losses[i]:.3f}")

# 检查最后10个loss
print(f"\n最后10个 loss:")
for i in range(max(0, len(losses)-10), len(losses)):
    print(f"  Step {steps[i]}: {losses[i]:.3f}")

# 检查loss是否下降
if len(losses) > 1:
    first_100 = losses[:min(100, len(losses))]
    last_100 = losses[-min(100, len(losses)):]
    print(f"\n前100步平均loss: {sum(first_100)/len(first_100):.3f}")
    print(f"后100步平均loss: {sum(last_100)/len(last_100):.3f}")
    print(f"变化: {sum(last_100)/len(last_100) - sum(first_100)/len(first_100):.3f}")


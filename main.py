import torch
from PIL import Image
from extraProcessing import load_image, save_image
from modelProcessing import StyleTransferModel, content_loss, style_loss

# ========== 所有可配置的参数和路径 ==========

# 图片路径
CONTENT_PATH = 'image/content.png'
STYLE_PATH   = 'image/style.jpg'
OUTPUT_PATH  = 'output1/output.jpg'
PROGRESSING_SHOW_PATH = 'output1/round'

# 记录原始内容图尺寸，用于最终输出还原
ORIG_SIZE = Image.open(CONTENT_PATH).size  # (宽, 高)

# 超参数
IMG_SIZE      = 512           # 图片尺寸（越大越清晰，越慢）
STEPS         = 700           # 迭代次数（越多风格越强）
STYLE_WEIGHT  = 1_000_000     # 风格权重（越大越像风格图）
LR            = 0.2           # Adam 学习率
SHOW_EVERY    = 100            # 每多少步保存一张中间结果

# ========== 加载图片 ==========
print("加载图片...")
content = load_image(CONTENT_PATH, size=IMG_SIZE)
style   = load_image(STYLE_PATH, size=IMG_SIZE)

# ========== 初始化生成图 ==========
generated = content.clone()
generated.requires_grad_(True)

# ========== 加载 VGG19（参数冻结） ==========
model = StyleTransferModel()
model.eval()

# ========== 预计算内容特征和风格特征（只算一次） ==========
print("提取内容特征和风格特征...")
with torch.no_grad():
    content_feat = model(content)
    style_feat   = model(style)

# ========== 配置优化器 ==========
optimizer = torch.optim.Adam([generated], lr=LR)

# ========== 主循环 ==========
print(f"\n开始迭代，共 {STEPS} 步...\n")

for step in range(STEPS):

    optimizer.zero_grad() #清理梯度

    gen_feat = model(generated)

    loss_c = content_loss(gen_feat[4], content_feat[4])

    loss_s = 0
    for i in range(5):
        loss_s += style_loss(gen_feat[i], style_feat[i])

    total_loss = loss_c + STYLE_WEIGHT * loss_s
    total_loss.backward()
    optimizer.step()

    if (step + 1) % SHOW_EVERY == 0:
        print(f"Step {step+1:>4d}: 总损失 = {total_loss.item():.2f}")
        save_image(generated, f'{PROGRESSING_SHOW_PATH}_{step+1}.jpg')

# ========== 保存最终结果 ==========
print("\n完成！保存最终结果...")
save_image(generated, OUTPUT_PATH, target_size=ORIG_SIZE)
print(f"风格迁移完成！请查看 {OUTPUT_PATH}")
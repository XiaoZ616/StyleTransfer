# StyleTransfer

这是一个基于 PyTorch 和 Torchvision 的神经风格迁移项目。项目使用预训练 VGG19 网络提取图像特征，将内容图像的结构与风格图像的纹理、色彩和笔触进行融合，生成新的风格化图像。

## 项目结构

```text
.
+-- main.py                 # 主程序，执行风格迁移
+-- modelProcessing.py      # VGG19 特征提取模型和损失函数
+-- extraProcessing.py      # 图像读取、预处理和保存工具
+-- test.py                 # 输出图像尺寸调整辅助脚本
+-- requirement.txt         # Python 依赖列表
`-- image/                  # 输入图片目录
    +-- content.png
    +-- content2.jpg
    `-- style.jpg
```

默认情况下，生成结果会保存到 `output1/` 目录中；中间过程图像会按 `output1/round_<step>.jpg` 的格式保存。

## 环境要求

建议使用 Python 3.10 或更高版本。

主要依赖：

- PyTorch
- Torchvision
- Pillow
- NumPy
- Matplotlib

安装依赖：

```bash
pip install -r requirement.txt
```

如果需要使用 GPU 加速，请根据自己的 CUDA 版本安装对应的 PyTorch 版本。

## 使用方法

1. 将内容图和风格图放入 `image/` 目录。
2. 根据需要修改 `main.py` 顶部的路径和参数：

```python
CONTENT_PATH = 'image/content.png'
STYLE_PATH = 'image/style.jpg'
OUTPUT_PATH = 'output1/output.jpg'
```

3. 运行程序：

```bash
python main.py
```

第一次运行时，Torchvision 会自动下载预训练的 VGG19 权重。程序运行完成后，最终结果会保存到 `output1/output.jpg`。

## 主要参数

- `IMG_SIZE`：处理图像的尺寸。数值越大，结果越清晰，但运行更慢、占用内存更多。
- `STEPS`：优化迭代次数。数值越大，风格融合通常越充分。
- `STYLE_WEIGHT`：风格损失权重。数值越大，生成图越接近风格图。
- `LR`：Adam 优化器学习率。
- `SHOW_EVERY`：每隔多少步保存一次中间结果。

## 说明

- 当前代码默认在 CPU 上运行；如需 GPU 加速，可以进一步将模型和张量移动到 CUDA 设备。
- `output/`、`output1/`、虚拟环境、缓存文件和个人笔记已在 `.gitignore` 中忽略。
- `image/` 目录中的图片会被 Git 跟踪，方便保留示例输入。上传公开仓库前，请确认这些图片没有隐私或版权问题。

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

def load_image(image_path, size=512):
    """
    加载图片并预处理为张量
    """
    # 打开图片并确保 RGB 三通道
    img = Image.open(image_path).convert('RGB')
    
    # 缩放尺寸
    img = img.resize((size, size))
    
    # 转为 numpy 数组，形状 (H, W, C)，值范围 0~255
    img_np = np.array(img, dtype=np.float32)
    
    # 归一化到 [0, 1]
    img_np = img_np / 255.0
    
    # 标准化：减均值，除标准差（ImageNet 标准值）
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    # 每个通道减去自己的均值，再除以自己的标准差
    img_np = (img_np - mean) / std
    
    # 转换维度：Pytorch 要求 (C, H, W)，而当前是 (H, W, C)
    img_np = img_np.transpose((2, 0, 1))   # 将通道维度移到前面
    
    # 转为 torch 张量，并添加 batch 维度 (1, C, H, W)
    img_tensor = torch.from_numpy(img_np).float().unsqueeze(0).contiguous()
    
    return img_tensor

def save_image(tensor, save_path, target_size=None):
    """将处理后的张量还原保存为图片，target_size=(宽, 高) 可恢复原图尺寸"""
    
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    
    img = tensor.squeeze(0)
    img = img * std + mean #逆标准化
    img = img.clamp(0, 1) #截断在0-1之间
    
    to_pil=transforms.ToPILImage() #转换器
    pil_img=to_pil(img) #转换为PIL图片

    if target_size is not None:
        pil_img = pil_img.resize(target_size, Image.LANCZOS)
    
    pil_img.save(save_path)
    
    print(f"已保存: {save_path}")
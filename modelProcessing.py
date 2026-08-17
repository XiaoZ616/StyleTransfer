import torch.nn as nn
from torchvision import models
import torch

class StyleTransferModel(nn.Module):
    def __init__(self):
        super().__init__()

        # VGG19 预训练模型的特征提取部分
        self.model = models.vgg19(pretrained=True).features

        # 只取6层
        self.layers = ['0', '5', '10', '19', '21'] # ‘21’层作为内容特征层

    def forward(self, x):
        """
        输入图片 → 输出 6 个特征图
        """
        features = []
        for layer_num, layer in self.model.named_children(): #named_children会返回一个迭代器，每次返回 (层号, 层对象) 的元组。
            x = layer(x)
            if layer_num in self.layers:
                features.append(x)
        return features

# ------ 损失函数--------

def content_loss(feat_generated, feat_content):
    """
    内容损失
    mse
    feat_generated:生成图片的内容特征图
    feat_content:原图片的内容特征图
    """
    diff = feat_generated - feat_content
    return torch.mean(diff ** 2)

def gram_matrix(feat):
    """
    Gram 矩阵计算
    输入: [1, C, H, W]特征图
    输出: [C, C]，归一化
    """
    _, c, h, w = feat.shape
    f = feat.view(1, c, h * w).squeeze(0)   # [C, H*W]
    gram = f @ f.T                           # [C, C]
    #gram = gram.unsqueeze(0)                 # [1, C, C]
    return gram / (c * h * w)    

def style_loss(feat_generated, feat_style):
    """
    风格损失
    torch.mean(...) 对所有元素取平均值，得到一个标量值
    """
    G_g = gram_matrix(feat_generated)
    G_s = gram_matrix(feat_style)
    diff = G_g - G_s
    return torch.mean(diff ** 2)
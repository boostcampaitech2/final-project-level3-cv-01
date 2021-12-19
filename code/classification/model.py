import sys
import timm
import torch.nn as nn

from models.yolo import Model

class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.name = 'tf_efficientnetv2_b2'
        self.model = timm.create_model(self.name, num_classes=num_classes, pretrained=True)

    def forward(self, X):
        return self.model(X)

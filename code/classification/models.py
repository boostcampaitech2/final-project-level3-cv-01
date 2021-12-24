import timm
import torch.nn as nn

class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.name = 'tf_efficientnetv2_b2'
        self.model = timm.create_model(self.name, num_classes=num_classes, pretrained=True)

    def forward(self, X):
        return self.model(X)


class Model(nn.Module):
    def __init__(self, name, num_classes=4, pretrained=True):
        super().__init__()
        self.name = name
        self.model = timm.create_model(self.name, num_classes=num_classes, pretrained=pretrained)

    def forward(self, x):
        return self.model(x)
from classification.model import EfficientNetV2

def train():
    cls_model = EfficientNetV2(num_classes=4)
    # TODO : train classification model
    # 1. create loader
    # 2. crop images with GT
    # 3. train
    # 4. comput loss
    # 5. validation
    # 6. save checkpoint

if __name__ == "__main__":
    train()
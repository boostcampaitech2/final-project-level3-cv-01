import time
import yaml
import argparse
import os.path as osp
import wandb
from tqdm import tqdm

import torch
import numpy as np

from classification.model import EfficientNetV2
from classification.datasets import create_dataloader
from classification.metrics import compute_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Train Segmentation Model')

    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('ckpt', type=str, help='checkpoint file path')
    parser.add_argument('--num-classes', type=int, default=4, help='number of classes')
    parser.add_argument('--img-size', type=int, default=1280, help='train image size')
    parser.add_argument('--entity', type=str, default='perforated_line', help='wandb entity name')
    parser.add_argument('--name', type=str, default='test', help='test name')
    
    args = parser.parse_args()

    return args


def test(path, args, device):
    test_loader = create_dataloader(path, args.img_size, 1, workers=8)
    
    model = EfficientNetV2(num_classes=4).to(device)
    model.load_state_dict(torch.load(args.ckpt))
    model.eval()

    pred_list = []
    gt_list = []
    mean_time = 0

    pbar = tqdm(enumerate(test_loader), total=len(test_loader))
    with torch.no_grad():
        for _, (image, label) in pbar:
            image = image.to(device).float() / 255.0
            label = label.to(device)

            st = time.time()
            output = model(image)
            et = time.time()

            pred = output.argmax(dim=-1)

            pred_list.append(pred)
            gt_list.append(label)
            mean_time += et - st

    mean_time /= len(test_loader)
    print(f"mean inference time : {mean_time}")

    metrics = compute_metrics(pred_list, gt_list, num_classes=args.num_classes)

    metric_names = ['f1', 'precision', 'recall']
    class_names = ['AH', 'A~H', '~AH', '~A~H']
    mean_metrics = []
    tags = []

    print(('%10s' * 5) % ("Result", *class_names))
    for metric_name, value in zip(metric_names, metrics):
        print(('%10s' + '%10.3f' * 4) % (metric_name, *value))
        mean_metrics.append(value.mean())
        tags.extend(['/'.join([metric_name, class_name]) for class_name in class_names])

    wandb.log({tag: metric for tag, metric in zip(tags, np.stack(metrics).reshape(-1))})

if __name__ == "__main__":
    args = parse_args()
    config = args.config

    with open(args.config) as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    wandb.init(config=args, entity=args.entity,
                project='YOLOR_classification',
                name=args.name)

    test(cfg_dict['test'], args, device)

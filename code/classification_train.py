import yaml
import argparse
import wandb
import os
import os.path as osp

import math
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from typing import Dict

from classification.datasets import create_dataloader
from classification.models import EfficientNetV2, Model


def parse_args():
    parser = argparse.ArgumentParser(description='Train Classification Model')

    parser.add_argument('config', type=str, help='config file path')
    parser.add_argument('--model', type=str, default='resnet101', help='model name')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')    
    parser.add_argument('--img-size', type=int, default=1280, help='train image size')
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--entity', type=str, default='perforated_line', help='wandb entity name')
    parser.add_argument('--save-dir', type=str, default='runs/classificaiton', help='directory to save model')
    parser.add_argument('--name', type=str, default='exp', help='experiment name')
    
    args = parser.parse_args()

    return args


def train(args:argparse.Namespace,
        epochs:int, cfg_dict:Dict,
        device:torch.device) -> None:

    save_dir = osp.join(args.save_dir, args.name, 'weights')
    os.makedirs(save_dir, exist_ok=True)

    acc_name = osp.join(save_dir, 'best_acc.pt')
    loss_name = osp.join(save_dir, 'best_loss.pt')

    train_loader = create_dataloader(cfg_dict['train'], args.img_size, args.batch_size, workers=8)
    valid_loader = create_dataloader(cfg_dict['valid'], args.img_size, args.batch_size, workers=8)

    model = Model(name=args.model, num_classes=4, pretrained=True).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg_dict['lr'], momentum=cfg_dict['momentum'])

    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - cfg_dict['lrf']) + cfg_dict['lrf']  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = -1.0
    best_loss = 1e6

    for i in range(epochs):
        epoch = i+1
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        # Train
        print(f"Epoch : {epoch}/{epochs}")
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for _, (images, labels) in pbar:
            images = images.to(device).float() / 255.0
            labels = labels.to(device).long()

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (preds == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)

        # scheduler.step()

        print(f"train accuracy : {train_acc:7.3f} | train loss : {train_loss:7.3f}")
        wandb.log({'train/accuracy': train_acc, 'train/loss': train_loss, 'z(etc)/lr':optimizer.param_groups[0]['lr']})

        # Validation
        with torch.no_grad():
            model.eval()
            valid_loss = 0.0
            valid_acc = 0.0

            for images, labels in valid_loader:
                images = images.to(device).float() / 255.0
                labels = labels.to(device).long()

                outputs = model(images)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                valid_loss += loss.item()
                valid_acc += (preds == labels).sum().item()

            valid_loss /= len(valid_loader)
            valid_acc /= len(valid_loader.dataset)
            
            print(f"valid accuracy : {valid_acc:7.3f} | valid loss : {valid_loss:7.3f}")
            wandb.log({'valid/accuracy': valid_acc, 'valid/loss': valid_loss})

            if best_acc < valid_acc:
                best_acc = valid_acc
                torch.save(model.state_dict(), acc_name)

            if best_loss > valid_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), loss_name)


if __name__ == "__main__":
    args = parse_args()
    epochs = args.epochs

    with open(args.config) as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    wandb.init(config=args, entity=args.entity,
                project='YOLOR_classification',
                name=args.name)

    train(args, epochs, cfg_dict, device)
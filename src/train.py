
#!/usr/bin/env python3
import yaml
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import argparse
from src.models.unet import ResNetUNet
from src.datasets import SegmentationDataset
import numpy as np

def train(cfg, epochs=5, device='cpu'):
    train_dir = cfg['data']['train_dir']
    img_dir = train_dir
    mask_dir = train_dir
    ds = SegmentationDataset(img_dir, mask_dir, size=tuple(cfg['data']['input_size']))
    dl = DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=0)
    model = ResNetUNet(n_classes=1, pretrained=False).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg['training']['lr'])
    loss_fn = nn.BCELoss()
    os.makedirs(cfg['paths']['checkpoint_dir'], exist_ok=True)
    for ep in range(epochs):
        model.train()
        loop = tqdm(dl, desc=f"Epoch {ep+1}/{epochs}")
        epoch_loss = 0.0
        for imgs, masks in loop:
            imgs = torch.from_numpy(np.array(imgs)).float().to(device) if isinstance(imgs, np.ndarray) else imgs.to(device).float()
            masks = torch.from_numpy(np.array(masks)).float().to(device) if isinstance(masks, np.ndarray) else masks.to(device).float()
            imgs = imgs.float()
            masks = masks.float()
            preds = model(imgs)
            loss = loss_fn(preds, masks)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        torch.save(model.state_dict(), os.path.join(cfg['paths']['checkpoint_dir'], f'model_epoch_{ep+1}.pth'))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/config.yaml')
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault('paths', {})
    cfg['paths'].setdefault('checkpoint_dir', 'runs')
    train(cfg, epochs=args.epochs, device=args.device)

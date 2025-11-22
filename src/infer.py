
#!/usr/bin/env python3
import argparse
import torch
import cv2
import numpy as np
from src.models.unet import UNetTiny
from src.preproc import preprocess_image

def infer(image_path, weights, out_path='out.png', device='cpu'):
    model = UNetTiny().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    img = cv2.imread(image_path)
    proc = preprocess_image(img)
    inp = torch.from_numpy(proc).unsqueeze(0).to(device).float()
    with torch.no_grad():
        pred = model(inp)[0,0].cpu().numpy()
    mask = (pred>0.5).astype('uint8')*255
    mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    img_resized = cv2.resize(img, (mask.shape[1], mask.shape[0]))
    overlay = cv2.addWeighted(img_resized, 0.7, mask_color, 0.3, 0)
    cv2.imwrite(out_path, overlay)
    print('Saved', out_path)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--out', default='out.png')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    infer(args.image, args.weights, args.out, args.device)

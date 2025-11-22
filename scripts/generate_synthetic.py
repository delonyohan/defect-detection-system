
#!/usr/bin/env python3
"""Generate synthetic metal textures with random scratches (simple overlays)."""
import os
import cv2
import numpy as np
from pathlib import Path
import argparse

def random_metal(h, w):
    base = np.random.normal(loc=120, scale=10, size=(h, w)).astype(np.uint8)
    # subtle vertical grain
    for i in range(0, w, 20):
        delta = np.random.randint(-5, 6)
        base[:, i:i+2] = np.clip(base[:, i:i+2] + delta, 0, 255)
    return cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

def add_scratch(img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    # random polyline
    num_pts = np.random.randint(5, 12)
    xs = np.linspace(int(w*0.1), int(w*0.9), num=num_pts).astype(int)
    ys = (np.sin(np.linspace(0, np.pi*2, num=num_pts)) * (h*0.02) + 
          np.random.randint(h//3, 2*h//3, size=num_pts)).astype(int)
    pts = np.vstack([xs, ys]).T
    thickness = np.random.randint(1, 4)
    cv2.polylines(mask, [pts], False, 255, thickness=thickness)
    # blurred center line
    scratch = cv2.GaussianBlur(mask, (5,5), 0)
    colored = img.copy()
    colored[scratch>0] = np.clip(colored[scratch>0] + 40, 0, 255)
    rim = cv2.dilate(mask, np.ones((3,3), np.uint8)) - mask
    colored[rim>0] = np.clip(colored[rim>0] - 30, 0, 255)
    return colored, mask

def main(out_dir, count=200, size=512):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for i in range(count):
        img = random_metal(size, size)
        img_s, mask = add_scratch(img)
        cv2.imwrite(os.path.join(out_dir, f"img_{i:04d}.png"), img_s)
        cv2.imwrite(os.path.join(out_dir, f"mask_{i:04d}.png"), mask)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir', default='data/synthetic')
    p.add_argument('--count', type=int, default=200)
    p.add_argument('--size', type=int, default=512)
    args = p.parse_args()
    main(args.out_dir, args.count, args.size)

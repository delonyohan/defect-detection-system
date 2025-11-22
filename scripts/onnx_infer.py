
#!/usr/bin/env python3
import onnxruntime as ort
import numpy as np
import cv2
from src.preproc import preprocess_image
import argparse

def run_infer(onnx_path, image_path, out_path=None):
    sess = ort.InferenceSession(onnx_path)
    img = cv2.imread(image_path)
    proc = preprocess_image(img)
    inp = np.expand_dims(proc, 0).astype(np.float32)
    out = sess.run(None, {'input': inp})[0]
    pred = out[0,0]
    mask = (pred>0.5).astype('uint8')*255
    mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.resize(img, (mask.shape[1], mask.shape[0])), 0.7, mask_color, 0.3, 0)
    if out_path:
        cv2.imwrite(out_path, overlay)
        print('Saved overlay to', out_path)
    return overlay

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--out', default=None)
    args = p.parse_args()
    run_infer(args.onnx, args.image, args.out)

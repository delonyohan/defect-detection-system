
#!/usr/bin/env python3
"""Export a trained PyTorch ResNetUNet model to ONNX and run a quick verification with ONNX Runtime."""
import argparse
import torch
import onnx
import onnxruntime as ort
from src.models.unet import ResNetUNet
import numpy as np

def export(weights, out_onnx, input_size=(3,512,512), device='cpu'):
    model = ResNetUNet(n_classes=1, pretrained=False).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    dummy = torch.randn(1, *input_size).to(device)
    torch.onnx.export(
        model,
        dummy,
        out_onnx,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input':{0:'batch'}, 'output':{0:'batch'}}
    )
    print('Exported ONNX model to', out_onnx)
    # verify with onnxruntime
    sess = ort.InferenceSession(out_onnx)
    inp = np.random.randn(1, *input_size).astype(np.float32)
    out = sess.run(None, {'input': inp})
    print('ONNX Runtime output shape:', [o.shape for o in out])

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--out', default='runs/model.onnx')
    args = p.parse_args()
    export(args.weights, args.out)

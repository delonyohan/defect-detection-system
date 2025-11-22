
#!/usr/bin/env python3
"""Simple dynamic quantization for ONNX using onnxruntime.quantization utilities."""
import argparse
from onnxruntime.quantization import quantize_dynamic, QuantType

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', required=True)
    parser.add_argument('--out', default='runs/model.quant.onnx')
    args = parser.parse_args()
    quantize_dynamic(args.onnx, args.out, weight_type=QuantType.QINT8)
    print('Saved quantized model to', args.out)

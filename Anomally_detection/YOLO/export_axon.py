"""Export the Axon YOLO26 variant (Configs/yolo26n_axon.yaml) to a full-INT8 .tflite for the nRF54LM20 Axon NPU.

python YOLO/export_axon.py --weights runs/yolo26n_axon/weights/best.pt

The fused PyTorch model is rebuilt layer by layer in Keras (NHWC) with the same weights, then converted like the
classifier (Classifier/train.py): full INT8, int8 input and outputs. ONNX -> onnx2tf mis-reads the C3k2 channel
split, so it is not used. The graph stops at the one-to-one head's six convolutions (box and class logit per
level, P3/P4/P5): only Axon ops, plus nearest upsampling, which the Axon runtime runs on the CPU. Box decode,
sigmoid and top-k run on the board CPU, as runner.Detector(backend="tflite") does on the PC. Each head conv is its
own output with its own int8 scale (one scale for boxes and scores rounds every score away, see export.py).

Writes next to the weights: best_axon_int8.tflite and axon_int8.h (input size, strides, input LUT).
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import tf_keras as keras
import torch
import yaml
from tf_keras import layers
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules import SPPF, Bottleneck, C2f, C3, Concat, Conv, DWConv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import decode_axon  # noqa: E402
from src import DEFAULT_CONFIG, resolve  # noqa: E402


def k_conv(x, m):
    """Fused ultralytics Conv / DWConv (conv with bias + ReLU or no act) or a plain nn.Conv2d, NHWC."""
    c = m.conv if isinstance(m, Conv) else m
    w, b = c.weight.detach().numpy(), c.bias.detach().numpy()
    k, s, p = c.kernel_size[0], c.stride[0], c.padding[0]
    if s > 1 and p:  # PyTorch pads both sides; TF 'same' at stride 2 pads only the bottom/right
        x, pad = layers.ZeroPadding2D(p)(x), "valid"
    else:
        pad = "same" if p else "valid"
    if c.groups == 1:
        x = layers.Conv2D(c.out_channels, k, s, pad, weights=[w.transpose(2, 3, 1, 0), b])(x)
    elif c.groups == c.in_channels == c.out_channels:
        x = layers.DepthwiseConv2D(k, s, pad, weights=[w.transpose(2, 3, 0, 1), b])(x)
    else:
        raise NotImplementedError(f"grouped conv {c}")
    act = getattr(m, "act", nn.Identity())
    if isinstance(act, nn.ReLU):
        return layers.ReLU()(x)
    if not isinstance(act, nn.Identity):
        raise NotImplementedError(f"activation {act}: Axon needs ReLU (set activation: nn.ReLU() in the yaml)")
    return x


def cat(ts):
    """Channel concat as a chain of 2-input concats: the Axon compiler mis-reads 3 or more inputs (error -216/-235)."""
    x = ts[0]
    for t in ts[1:]:
        x = layers.Concatenate()([x, t])
    return x


def k_block(x, m):
    """One ultralytics module on a Keras tensor (x is a list for Concat)."""
    if isinstance(m, (Conv, DWConv, nn.Conv2d)):
        return k_conv(x, m)
    if isinstance(m, nn.Sequential):
        for sub in m:
            x = k_block(x, sub)
        return x
    if isinstance(m, Bottleneck):
        y = k_conv(k_conv(x, m.cv1), m.cv2)
        return layers.Add()([x, y]) if m.add else y
    if isinstance(m, C2f):  # also C3k2
        y = k_conv(x, m.cv1)
        y = [y[..., :m.c], y[..., m.c:]]
        y.extend(k_block(y[-1], b) for b in m.m)
        return k_conv(cat(y), m.cv2)
    if isinstance(m, C3):  # also C3k
        return k_conv(cat([k_block(k_conv(x, m.cv1), m.m), k_conv(x, m.cv2)]), m.cv3)
    if isinstance(m, SPPF):
        y = [k_conv(x, m.cv1)]
        y.extend(layers.MaxPooling2D(m.m.kernel_size, 1, "same")(y[-1]) for _ in range(m.n))
        y = k_conv(cat(y), m.cv2)
        return layers.Add()([x, y]) if m.add else y
    if isinstance(m, nn.Upsample):
        return layers.UpSampling2D(int(m.scale_factor), interpolation="nearest")(x)
    if isinstance(m, Concat):
        return cat(x)
    raise NotImplementedError(type(m).__name__)


def to_keras(model, size):
    """Backbone + neck + head convs -> Keras model with outputs [box P3, cls P3, box P4, cls P4, box P5, cls P5]."""
    inp = keras.Input((size, size, 3), batch_size=1)
    x, y, det = inp, [], model.model[-1]
    for m in model.model[:-1]:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = k_block(x, m)
        y.append(x)
    h = det.one2one if det.one2one_cv2 is not None else det.one2many  # fuse() keeps only the head in use
    outs = [k_block(y[j], hd[i]) for i, j in enumerate(det.f) for hd in (h["box_head"], h["cls_head"])]
    return keras.Model(inp, outs)


def torch_raw(model, x):
    """Same six outputs from the PyTorch model (NCHW), for the conversion check."""
    y, det = [], model.model[-1]
    for m in model.model[:-1]:
        if m.f != -1:
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = m(x)
        y.append(x)
    h = det.one2one if det.one2one_cv2 is not None else det.one2many
    return [hd[i](y[j]) for i, j in enumerate(det.f) for hd in (h["box_head"], h["cls_head"])]


def calib_images(cfg, n, size):
    train = sorted((resolve(cfg["paths"]["dataset"]) / "images" / "train").glob("*"))
    for p in random.Random(cfg["train"]["seed"]).sample(train, min(n, len(train))):
        img = cv2.resize(cv2.imread(str(p)), (size, size), interpolation=cv2.INTER_LINEAR)
        yield img[None, ..., ::-1].astype(np.float32) / 255.0  # NHWC RGB 0..1


def write_header(path, interp, size, strides):
    scale, zp = interp.get_input_details()[0]["quantization"]
    lut = [int(np.clip(round(v / 255.0 / scale + zp), -128, 127)) for v in range(256)]
    path.write_text(
        "/* Generated by YOLO/export_axon.py. Input: RGB uint8 -> int8 through det_input_lut, CHW for Axon. */\n"
        "#pragma once\n#include <stdint.h>\n"
        f"#define DET_INPUT_SIZE {size}\n#define DET_LEVELS {len(strides)}\n"
        f"static const uint8_t det_strides[DET_LEVELS] = {{{', '.join(map(str, strides))}}};\n"
        "static const int8_t det_input_lut[256] = {\n"
        + ",\n".join("\t" + ", ".join(map(str, lut[i:i + 16])) for i in range(0, 256, 16)) + "\n};\n")


def export(cfg, weights, size, n_calib):
    model = YOLO(str(weights)).model.float().eval()
    model.model[-1].end2end = model.yaml.get("end2end", False)  # as in ultralytics' own export: one-to-one head
    model.fuse()
    strides = [int(s) for s in model.model[-1].stride]
    km = to_keras(model, size)

    x = next(calib_images(cfg, 1, size))
    with torch.no_grad():
        ref = [t.numpy().transpose(0, 2, 3, 1) for t in torch_raw(model, torch.from_numpy(x.transpose(0, 3, 1, 2)))]
    kout = km.predict(x, verbose=0)
    err = max(float(np.abs(a - b).max()) for a, b in zip(kout, ref))
    assert err < 1e-3 * max(float(np.abs(r).max()) for r in ref), f"Keras rebuild differs from PyTorch by {err}"
    # the board's decode (runner.decode_axon) must give ultralytics' own end-to-end top box
    with torch.no_grad():
        e2e = model(torch.from_numpy(x.transpose(0, 3, 1, 2)))[0][0, 0, :5].numpy()
    ours = decode_axon(kout, size, 0.0, 1)[0, 0, :5]
    assert np.allclose(ours, e2e, atol=1e-2), f"decode mismatch: ours {ours}, ultralytics {e2e}"

    conv = tf.lite.TFLiteConverter.from_keras_model(km)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: ([x] for x in calib_images(cfg, n_calib, size))
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = conv.inference_output_type = tf.int8
    tfl = weights.with_name("best_axon_int8.tflite")
    tfl.write_bytes(conv.convert())

    interp = tf.lite.Interpreter(model_path=str(tfl))
    ops = sorted({d["op_name"] for d in interp._get_ops_details()})
    write_header(weights.with_name("axon_int8.h"), interp, size, strides)
    print(f"[export_axon] Keras vs PyTorch max abs diff {err:.2e}")
    print(f"[export_axon] {tfl} ({tfl.stat().st_size / 1024:.0f} KB), ops: {', '.join(ops)}")
    for o in interp.get_output_details():
        print(f"  output {o['name']}: {o['shape'].tolist()} scale {o['quantization'][0]:.5f} zp {o['quantization'][1]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--imgsz", type=int, default=160)
    ap.add_argument("--calib", type=int, default=200, help="train images for INT8 calibration")
    a = ap.parse_args()
    export(yaml.safe_load(a.config.read_text()), a.weights.resolve(), a.imgsz, a.calib)


if __name__ == "__main__":
    main()

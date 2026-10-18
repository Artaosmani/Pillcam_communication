"""CPU latency of the exported YOLO26 model per format and thread count, timed per stage.

python YOLO/bench.py --weights runs/yolo26n/weights/best.pt [--threads 1 2 4] [--runs 200] [--warmup 20]

Input = the real pipeline input: test polyp images squashed to 336×336 BGR frames, decoded before timing (decode is
not part of the YOLO stage). For each format (onnx_fp32, onnx_int8, openvino_fp16 = the FP32 IR, which OpenVINO runs in
FP16 on ARM CPUs by default, openvino_int8 = the QDQ INT8 ONNX compiled by OpenVINO) and thread count: `warmup` untimed frames, then `runs` timed frames, one at a time (batch 1).
Stages: pre (letterbox + normalise), infer, post (threshold/NMS + rescale), total. Reports p50/p95/mean in ms.
Run on an otherwise idle machine. Writes <out>/yolo_bench.json and .csv (default out: results/).
"""
import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import Detector  # noqa: E402
from src import DEFAULT_CONFIG, resolve  # noqa: E402

FRAME = 336


def machine():
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return {"cpu": cpu or platform.processor(), "platform": platform.platform(), "python": platform.python_version()}


def stats(v):
    v = np.asarray(v) * 1e3
    return {"p50_ms": float(np.percentile(v, 50)), "p95_ms": float(np.percentile(v, 95)), "mean_ms": float(v.mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--out", type=Path, default=Path("results"))
    a = p.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    w = a.weights
    formats = {"onnx_fp32": (w.with_suffix(".onnx"), None), "onnx_int8": (w.with_name(f"{w.stem}_int8.onnx"), None),
               "openvino_fp16": (w.with_name(f"{w.stem}_openvino_model"), None),
               "openvino_int8": (w.with_name(f"{w.stem}_int8.onnx"), "openvino")}
    conf = json.loads((w.parent / "deploy.json").read_text())["conf"] if (w.parent / "deploy.json").exists() else 0.25

    src = sorted((resolve(cfg["paths"]["source"]) / "images" / "test").glob("*.jpg"))
    frames = [cv2.resize(cv2.imread(str(f)), (FRAME, FRAME), interpolation=cv2.INTER_AREA) for f in src]
    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {"machine": machine(), "weights": str(w), "conf": conf, "frame": f"{FRAME}x{FRAME}",
              "runs": a.runs, "warmup": a.warmup, "batch": 1, "results": []}
    rows = []
    for name, (path, backend) in formats.items():
        if not path.exists():
            print(f"[bench] {name}: {path} missing, skipped")
            continue
        for t in a.threads:
            det = Detector(path, threads=t, conf=conf, backend=backend)
            for i in range(a.warmup):
                det(frames[i % len(frames)])
            times = {"pre": [], "infer": [], "post": [], "total": []}
            for i in range(a.runs):
                t0 = time.perf_counter()
                _, s = det(frames[i % len(frames)])
                times["total"].append(time.perf_counter() - t0)
                for k in ("pre", "infer", "post"):
                    times[k].append(s[k])
            r = {"format": name, "threads": t, **{k: stats(v) for k, v in times.items()}}
            result["results"].append(r)
            rows += [{"format": name, "threads": t, "stage": k, **r[k]} for k in times]
            print(f"[bench] {name:14s} {t} thr: total p50 {r['total']['p50_ms']:.2f} p95 {r['total']['p95_ms']:.2f} ms "
                  f"(infer p50 {r['infer']['p50_ms']:.2f})")
    (out / "yolo_bench.json").write_text(json.dumps(result, indent=2))
    with open(out / "yolo_bench.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["format", "threads", "stage", "p50_ms", "p95_ms", "mean_ms"])
        wr.writeheader()
        wr.writerows(rows)
    print(f"[bench] -> {out}/yolo_bench.json, yolo_bench.csv")


if __name__ == "__main__":
    main()

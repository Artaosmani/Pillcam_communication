"""Time per stage of detection (YOLO26n, OpenVINO on the PC CPU, as in the pipeline) and classification (board).

python YOLO/stage_profile.py [--weights runs/yolo26n/weights/best.pt] [--threads 4] [--runs 300]

Detection, per frame (336×336 test frames, like bench.py), split into:
  pre   resize (letterbox scale to the model input), pad (114 border), to tensor (BGR->RGB, HWC->CHW, /255)
  infer backbone (layers 0-10), neck (11-22), head (23, Detect incl. box decode), runtime (wall time not in any
        layer: input copy, scheduling), from OpenVINO's per-layer counters (PERF_COUNT on)
  post  confidence filter (+ cx,cy,w,h -> corners), NMS (OpenVINO export has the one-to-many head; 0 for the
        end-to-end ONNX head), rescale + clip to the frame, sort
Classification: the board's measured preprocessing + Axon inference from results/latency_summary.json (one stage each).
Writes results/stage_times.json and results/stage_times.png.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import Detector  # noqa: E402
from src import DEFAULT_CONFIG, ROOT, resolve  # noqa: E402

FRAME = 336
PARTS = [("pre", "resize"), ("pre", "pad"), ("pre", "to tensor"),
         ("infer", "backbone"), ("infer", "neck"), ("infer", "head"), ("infer", "runtime"),
         ("post", "conf filter"), ("post", "NMS"), ("post", "rescale + clip"), ("post", "sort")]
COLORS = {"pre": "#2a78d6", "infer": "#eb6834", "post": "#1baf7a"}  # dataviz reference palette, slots 1-3


def part(layer):
    return "backbone" if layer <= 10 else "neck" if layer <= 22 else "head"


def layer_times(req):
    """ms per model part from OpenVINO's counters. Fused Subgraph/Reorder nodes can carry another layer's name, so
    they go to the layer of the node that ran just before them (counters are in execution order)."""
    # ponytail: attribution heuristic for fused nodes; exact per-layer split needs an unfused profile (ORT)
    out, layer = {"backbone": 0.0, "neck": 0.0, "head": 0.0}, 0
    for p in req.profiling_info:
        m = re.search(r"model\.(\d+)", p.node_name)
        if m and p.node_type not in ("Subgraph", "Reorder"):
            layer = int(m.group(1))
        out[part(layer)] += p.real_time.total_seconds() * 1e3
    return out


def run_frame(det, img):
    """Detector.preprocess / _run / postprocess, step by step with a clock between steps. Returns (boxes, ms)."""
    t, c = {}, time.perf_counter
    t0 = c()
    h, w = img.shape[:2]
    r = min(det.size / h, det.size / w)
    nw, nh = round(w * r), round(h * r)
    px, py = (det.size - nw) / 2, (det.size - nh) / 2
    im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img
    t1 = c()
    top, left = round(py - 0.1), round(px - 0.1)
    im = cv2.copyMakeBorder(im, top, det.size - nh - top, left, det.size - nw - left, cv2.BORDER_CONSTANT,
                            value=(114, 114, 114))
    t2 = c()
    x = np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    t3 = c()
    out = det._run(x)
    t4 = c()
    o = out[0]
    if o.shape[-1] == 6:  # end-to-end head (ONNX export): no NMS
        b = o[o[:, 4] >= det.conf, :5]
        t5 = t5n = c()
    else:  # one-to-many head (OpenVINO export): 1×5×A cx,cy,w,h,conf -> threshold, xyxy, NMS
        p = o.T[o[4] >= det.conf]
        xyxy = np.c_[p[:, :2] - p[:, 2:4] / 2, p[:, :2] + p[:, 2:4] / 2]
        t5 = c()
        keep = cv2.dnn.NMSBoxes(np.c_[xyxy[:, :2], p[:, 2:4]].tolist(), p[:, 4].tolist(), det.conf, det.iou)
        b = np.c_[xyxy, p[:, 4]][np.asarray(keep, int).ravel()] if len(p) else np.zeros((0, 5))
        t5n = c()
    b = b.astype(np.float64)
    b[:, [0, 2]] = ((b[:, [0, 2]] - left) / r).clip(0, w)
    b[:, [1, 3]] = ((b[:, [1, 3]] - top) / r).clip(0, h)
    t6 = c()
    b = b[np.argsort(-b[:, 4])]
    t7 = c()
    ms = lambda a, z: (z - a) * 1e3
    t = {"resize": ms(t0, t1), "pad": ms(t1, t2), "to tensor": ms(t2, t3), "infer_wall": ms(t3, t4),
         "conf filter": ms(t4, t5), "NMS": ms(t5, t5n), "rescale + clip": ms(t5n, t6), "sort": ms(t6, t7)}
    layers = layer_times(det.req)
    t.update(layers, runtime=max(t["infer_wall"] - sum(layers.values()), 0.0))
    return b, t


def stats(v):
    v = np.asarray(v)
    return {"p50_ms": float(np.percentile(v, 50)), "p95_ms": float(np.percentile(v, 95)), "mean_ms": float(v.mean())}


def style(ax):
    ax.set_facecolor("#fcfcfb")
    ax.tick_params(colors="#52514e", labelsize=8, length=0)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.6), ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)


def figure(det_mean, cls, path):
    """Top: both models as stacked bars (where the time goes overall). Bottom: every detection step on its own bar."""
    fig, (top, bot) = plt.subplots(2, 1, figsize=(10, 6.4), dpi=200, gridspec_kw={"height_ratios": [1, 2.2]})
    fig.patch.set_facecolor("#fcfcfb")
    bars = [("Detection\nYOLO26n, PC CPU", [(g, n, det_mean[n]) for g, n in PARTS]),
            ("Classification\nMobileNetV1, board", [("pre", "pre", cls["pre"]), ("infer", "Axon inference", cls["infer"])])]
    for y, (_, segs) in enumerate(bars):
        x0 = 0.0
        for g, n, v in segs:
            top.barh(y, v, left=x0, height=0.6, color=COLORS[g], edgecolor="#fcfcfb", linewidth=2)
            if v > 1.6:  # label only segments wide enough to hold text
                top.text(x0 + v / 2, y, n, ha="center", va="center", color="white", fontsize=8)
            x0 += v
        top.text(x0 + 0.2, y, f"{x0:.1f} ms", va="center", fontsize=9, color="#0b0b0b")
    top.set_yticks(range(len(bars)), [b[0] for b in bars], fontsize=9, color="#52514e")
    top.invert_yaxis()
    top.set_xlim(0, max(sum(v for *_, v in s) for _, s in bars) * 1.12)
    top.set_title("Time per frame by stage", loc="left", fontsize=10, color="#0b0b0b")
    style(top)
    top.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLORS[g]) for g in COLORS],
               labels=["preprocessing", "inference", "postprocessing"], ncol=3, frameon=False, fontsize=8,
               loc="lower right", bbox_to_anchor=(1, 1))

    vals = [det_mean[n] for _, n in PARTS]
    bot.barh(range(len(PARTS)), vals, height=0.7, color=[COLORS[g] for g, _ in PARTS])
    for y, v in enumerate(vals):
        bot.text(v + 0.05, y, f"{v:.3f} ms" if v < 0.1 else f"{v:.2f} ms", va="center", fontsize=8, color="#0b0b0b")
    bot.set_yticks(range(len(PARTS)), [n for _, n in PARTS], fontsize=8.5, color="#52514e")
    bot.invert_yaxis()
    bot.set_xlim(0, max(vals) * 1.15)
    bot.set_xlabel("ms per frame", color="#52514e", fontsize=9)
    bot.set_title("Detection, step by step (YOLO26n, OpenVINO, 4 threads)", loc="left", fontsize=10, color="#0b0b0b")
    style(bot)
    fig.text(0.01, 0.005, "Detection: mean of the profiling run on the PC. Classification: board p50 (the board's "
             "times barely vary).", fontsize=7, color="#52514e")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, facecolor=fig.get_facecolor())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--weights", type=Path, default=ROOT / "runs/yolo26n/weights/best.pt")
    p.add_argument("--threads", type=int, default=4, help="the pipeline runs YOLO with 4 threads")
    p.add_argument("--runs", type=int, default=300)
    p.add_argument("--warmup", type=int, default=20)
    a = p.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    w = a.weights
    conf = json.loads((w.parent / "deploy.json").read_text())["conf"]
    det = Detector(w.with_name(f"{w.stem}_openvino_model"), threads=a.threads, conf=conf)
    # recompile with per-layer counters; same model, threads and hint as Detector
    import openvino as ov
    xml = next(w.with_name(f"{w.stem}_openvino_model").glob("*.xml"))
    det.req = ov.Core().compile_model(str(xml), "CPU", {"INFERENCE_NUM_THREADS": a.threads, "PERFORMANCE_HINT": "LATENCY",
                                                         "PERF_COUNT": True}).create_infer_request()
    det._run = lambda x: det.req.infer({0: x})[0].copy()

    src = sorted((resolve(cfg["paths"]["source"]) / "images" / "test").glob("*.jpg"))
    frames = [cv2.resize(cv2.imread(str(f)), (FRAME, FRAME), interpolation=cv2.INTER_AREA) for f in src]
    # check: the step-by-step version gives exactly Detector's boxes
    for f in frames[:10]:
        assert np.array_equal(run_frame(det, f)[0], det(f)[0]), "stage timing diverged from Detector"
    for i in range(a.warmup):
        run_frame(det, frames[i % len(frames)])
    times = {}
    for i in range(a.runs):
        _, t = run_frame(det, frames[i % len(frames)])
        t["pre"] = t["resize"] + t["pad"] + t["to tensor"]
        t["post"] = t["conf filter"] + t["NMS"] + t["rescale + clip"] + t["sort"]
        t["total"] = t["pre"] + t["infer_wall"] + t["post"]
        for k, v in t.items():
            times.setdefault(k, []).append(v)
    det_stats = {k: stats(v) for k, v in times.items()}

    s = json.loads((ROOT / "results/latency_summary.json").read_text())
    cls = {"pre": s["2a_board_preprocess"]["p50_ms"], "infer": s["2b_board_inference"]["p50_ms"]}
    out = {"detection": {"model": str(w), "backend": "OpenVINO (FP16 compute on ARM)", "threads": a.threads,
                         "frame": f"{FRAME}x{FRAME}", "input": det.size, "runs": a.runs, "perf_count": True,
                         "stages": det_stats},
           "classification": {"model": "MobileNetV1 0.25 INT8, 84x84, Axon NPU on nRF54LM20 DK",
                              "source": "results/latency_summary.json (board run, p50)", "pre_ms": cls["pre"],
                              "infer_ms": cls["infer"], "total_ms": cls["pre"] + cls["infer"]}}
    (ROOT / "results/stage_times.json").write_text(json.dumps(out, indent=2))
    figure({k: v["mean_ms"] for k, v in det_stats.items()}, cls, ROOT / "results/stage_times.png")

    print("| Stage | Step | p50 ms | p95 ms | mean ms |\n|---|---|---|---|---|")
    for g, n in PARTS:
        v = det_stats[n]
        print(f"| {g} | {n} | {v['p50_ms']:.3f} | {v['p95_ms']:.3f} | {v['mean_ms']:.3f} |")
    for k in ("pre", "infer_wall", "post", "total"):
        v = det_stats[k]
        print(f"| **{k}** | | {v['p50_ms']:.2f} | {v['p95_ms']:.2f} | {v['mean_ms']:.2f} |")
    print(f"\nclassification (board): pre {cls['pre']:.2f} + infer {cls['infer']:.2f} = {cls['pre'] + cls['infer']:.2f} ms")
    print("-> results/stage_times.json, results/stage_times.png")


if __name__ == "__main__":
    main()

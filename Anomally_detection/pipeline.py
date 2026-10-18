"""One command: frames -> classifier CSV -> YOLO box CSV -> bbox_replay -> latency figure + per-stage summary.

python pipeline.py                                      # PC stand-in for the board (classifier on the PC)
python pipeline.py --clf runs/pipeline/board.csv        # classifier CSV recorded from the board (Board/receiver.py)
python pipeline.py --yolo runs/yolo26n/weights/best.onnx --threads 4 --out runs/pipeline_fp32

Frames: Data/test/frames/frame_NNNNNN.jpg, 336×336 JPEG (q90, ~40 KB), the shared test split (Data/splits.json):
Kvasir-SEG/Kvasir test polyps + test negatives, squashed to 336×336 the way the classifier is trained. Built once,
with Data/test/labels.csv (frame_id,file,polyp,source) and Data/test/gt_boxes.csv (Kvasir-SEG boxes in 336 px).
Outputs in --out: classifier.csv, boxes.csv (+ .timing.csv), bbox_replay_out.csv, latency_breakdown.png/.pdf,
latency_stages.csv, latency_summary.json. Every stage in the summary is labelled measured, host estimate or placeholder.
"""
import argparse
import csv
import json
import os
import random
import shlex
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
PBS = ROOT.parent / "PBS"
PY = sys.executable
FRAME = 336
BUDGET_MS = 100.0


def make_frames(frames, seed=0):
    """Write the replay frames + labels.csv + gt_boxes.csv from the test split. Order is a seeded shuffle."""
    split = json.loads((ROOT / "Data/splits.json").read_text())["test"]
    random.Random(seed).shuffle(split)
    seg_labels = ROOT / "Data/kvasir-seg-split/labels/test"
    frames.mkdir(parents=True)
    with open(frames.parent / "labels.csv", "w", newline="") as fl, \
            open(frames.parent / "gt_boxes.csv", "w", newline="") as fg:
        wl, wg = csv.writer(fl), csv.writer(fg)
        wl.writerow(["frame_id", "file", "polyp", "source"])
        wg.writerow(["frame_id", "x_min", "y_min", "x_max", "y_max"])
        for fid, (path, y) in enumerate(split, 1):
            img = cv2.resize(cv2.imread(str(ROOT / path)), (FRAME, FRAME), interpolation=cv2.INTER_AREA)
            name = f"frame_{fid:06d}.jpg"
            cv2.imwrite(str(frames / name), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            wl.writerow([fid, name, y, path])
            lab = seg_labels / f"{Path(path).stem}.txt"
            if "kvasir-seg-split" in path and lab.exists():
                for _, cx, cy, w, h in np.loadtxt(lab, ndmin=2):
                    wg.writerow([fid, *(round(v * FRAME, 1) for v in (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))])
    print(f"[frames] {len(split)} frames -> {frames}")


def run(cmd, **kw):
    print("[run]", " ".join(map(str, cmd)))
    subprocess.run(list(map(str, cmd)), check=True, **kw)


def build_replay():
    """Compile bbox_replay with OpenMP (Homebrew libomp on macOS), as in PBS/README.md. Melwin's code is not touched."""
    exe = PBS / "drift/bbox_replay"
    src = [PBS / "drift" / f for f in ("bbox_replay.cpp", "src_drift/particle.cpp", "src_drift/channel.cpp")]
    if exe.exists() and exe.stat().st_mtime > max(f.stat().st_mtime for f in src):
        return exe
    omp = Path("/opt/homebrew/opt/libomp")
    flags = ["-Xpreprocessor", "-fopenmp", f"-I{omp}/include", f"-L{omp}/lib", "-lomp"] if omp.exists() else ["-fopenmp"]
    run(["clang++" if omp.exists() else "g++", "-O3", "-std=c++17", *flags, *src, "-o", exe])
    return exe


def pct(v):
    v = np.asarray(v, float)
    return {"p50_ms": float(np.percentile(v, 50)), "p95_ms": float(np.percentile(v, 95)), "n": int(len(v))} if len(v) \
        else {"p50_ms": None, "p95_ms": None, "n": 0}


def summary(out, board, yolo_model, threads):
    """Per-stage p50/p95 with where/how each number was measured."""
    clf = list(csv.DictReader(open(out / "classifier.csv")))
    tim = list(csv.DictReader(open(out / "boxes.timing.csv")))
    rep = list(csv.DictReader(open(out / "bbox_replay_out.csv")))
    det = [r for r in clf if int(r["detected"])]
    ms = lambda r, a, b: (int(r[b]) - int(r[a])) / 1e6
    host = "host estimate (PC TFLite interpreter, 1 thread, built-in kernels without XNNPACK; NOT the nRF54LM20 board)"
    usb = [int(r["usb_ns"]) / 1e6 for r in det if r.get("usb_ns")]
    s = {
        "1_frame_to_board": {**pct([int(r["usb_down_ns"]) / 1e6 for r in clf if r.get("usb_down_ns")]),
                             "label": "measured" if board else "placeholder",
                             "how": "USB CDC-ACM host->board, from Board/receiver.py" if board else
                             "not measured: needs the board (lab) or the pill radio link (real system)"},
        "2a_board_preprocess": {**pct([ms(r, "t_rx_board_ns", "t_infer_start_ns") for r in clf]),
                                "label": "measured" if board else "host estimate",
                                "how": "board timer" if board else "PC: JPEG decode + area resize + normalise"},
        "2b_board_inference": {**pct([ms(r, "t_infer_start_ns", "t_infer_end_ns") for r in clf]),
                               "label": "measured" if board else "host estimate", "how": "board cycle counter" if board else host},
        "3_board_to_pc": {**pct(usb), "label": "measured" if usb else "placeholder",
                          "how": "USB CDC-ACM board->host incl. frame bytes, Board/receiver.py" if usb else
                          "not measured on the PC-only run (0 in the timeline)"},
        "4_yolo_pc_cpu": {**pct([float(r["yolo_ms"]) for r in tim]), "label": "measured",
                          "how": f"{yolo_model.name} ({'OpenVINO, FP16 compute on ARM' if yolo_model.is_dir() else 'onnxruntime'}), "
                                 f"PC CPU, {threads} threads, pre+infer+post"},
        "4a_yolo_pre": pct([float(r["pre_ms"]) for r in tim]),
        "4b_yolo_infer": pct([float(r["infer_ms"]) for r in tim]),
        "4c_yolo_post": pct([float(r["post_ms"]) for r in tim]),
        "5_simulation_per_box": {**pct([float(r["latency_ms"]) for r in rep]), "label": "measured",
                                 "how": "bbox_replay process_box() wall time (full PBS run per box until the lookup table exists)"},
    }
    boxes = list(csv.DictReader(open(out / "boxes.csv")))
    first = {}
    for b in boxes:  # one detection latency per frame
        first.setdefault(b["frame_id"], ms(b, "t_capture_ns", "t_detect_ns"))
    total = pct(list(first.values()))
    s["detection_total"] = {**total, "budget_p95_ms": BUDGET_MS,
                            "within_budget": total["p95_ms"] is not None and total["p95_ms"] <= BUDGET_MS,
                            "label": "measured" if board else "host estimate",
                            "how": "t_detect - t_capture per frame with boxes = stage 2 + 3 + 4 (stage 1 not included)"}
    s["counts"] = {"frames": len(clf), "detected": len(det), "frames_with_boxes": len(first), "boxes": len(boxes)}
    (out / "latency_summary.json").write_text(json.dumps(s, indent=2))
    for k, v in s.items():
        if "p50_ms" in v and v["p50_ms"] is not None:
            print(f"[latency] {k:22s} p50 {v['p50_ms']:8.2f}  p95 {v['p95_ms']:8.2f} ms  {v.get('label', '')}")
    print(f"[latency] counts {s['counts']}")


def main():
    a = parse_args()
    out = a.out if a.out.is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    if not a.frames.exists():
        make_frames(a.frames)
    if a.clf:
        clf = a.clf
    else:
        clf = out / "classifier.csv"
        run([PY, ROOT / "Classifier/run_csv.py", "--frames", a.frames, "--model", a.classifier, "--out", clf,
             "--labels", a.frames.parent / "labels.csv"])
    if clf != out / "classifier.csv":
        (out / "classifier.csv").write_bytes(Path(clf).read_bytes())
    run([PY, ROOT / "YOLO/run_boxes.py", "--clf", out / "classifier.csv", "--frames", a.frames, "--model", a.yolo,
         "--threads", a.threads, "--out", out / "boxes.csv"])
    replay = build_replay()
    run([replay, "--csv", out / "boxes.csv", "--out", out / "bbox_replay_out.csv", *shlex.split(a.replay_args)])
    run([PY, PBS / "visualizers/plot_latency_breakdown.py", "--boxes", out / "boxes.csv",
         "--server", out / "bbox_replay_out.csv", "--out-dir", out, "--max-bars", 20,
         "--subtitle", "detection = measured (" + ("board" if a.clf else "PC stand-in for the board") +
         " + YOLO); capture and network still fake; lookup = full PBS run per box"],
        cwd=PBS / "visualizers")
    summary(out, bool(a.clf), a.yolo, a.threads)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", type=Path, default=ROOT / "Data/test/frames")
    p.add_argument("--classifier", type=Path, default=ROOT / "runs/classifier_84/model_int8.tflite")
    p.add_argument("--clf", type=Path, help="use this classifier CSV (e.g. from the board) instead of running it here")
    p.add_argument("--yolo", type=Path, default=ROOT / "runs/yolo26n/weights/best_openvino_model")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("runs/pipeline"))
    p.add_argument("--replay-args", default="", help='passed to bbox_replay, e.g. --replay-args="--rows 20"')
    return p.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()

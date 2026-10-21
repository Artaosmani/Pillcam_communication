"""Collect the run outputs into results/: RESULTS.md, the JSON/CSV behind it, the latency figure and example frames.

python report.py [--pipeline runs/pipeline] [--yolo runs/yolo26n runs/yolo26n_p2] [--classifier runs/classifier_96 ...]
                 [--board runs/pipeline_board runs/pipeline_board_raw]

Reads only files written by the other scripts (metrics_test.json, metrics_capsule.json, metrics_eval.json,
results/yolo_bench.json, <pipeline>/latency_summary.json, boxes.csv, <board run>/board.csv) and copies them into
results/.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def f(v, d=3):
    return "–" if v is None else f"{v:.{d}f}"


def classifier_section(runs):
    lines = ["## Classifier: MobileNetV1 0.25, binary polyp yes/no (board)", "",
             "Test split: Kvasir-SEG test polyps (+ their Kvasir v1 copies) and test negatives (HyperKvasir lower-GI "
             "normals, Kvasir-Capsule normal mucosa, Kvasir v1 ulcerative colitis / dyed resection margins). "
             "Threshold picked on val for ≥95 % recall.", "",
             "| Model | Input | Precision | Recall | F1 | Specificity | ROC-AUC | TN / FP / FN / TP | Threshold |",
             "|---|---|---|---|---|---|---|---|---|"]
    stats, capsule = [], []
    for run in runs:
        m = load(run / "metrics_test.json")
        if not m:
            continue
        size = m["model"]["imgsz"]
        for k in ("float", "int8"):
            r, cm = m[k], m[k]["confusion_matrix"]
            thr = f"logit ≥ {m['threshold']['float']['logit']:.2f}" if k == "float" else \
                f"int8 ≥ {m['threshold']['int8']['raw']}"
            lines.append(f"| {k.upper() if k == 'int8' else 'Float'} | {size}×{size} | {f(r['precision'])} | "
                         f"{f(r['recall'])} | {f(r['f1'])} | {f(r['specificity'])} | {f(r['roc_auc'])} | "
                         f"{cm['tn']} / {cm['fp']} / {cm['fn']} / {cm['tp']} | {thr} |")
        s = m["model"]
        stats.append(f"| {size}×{size} | {s['params']:,} | {s['macs'] / 1e6:.2f} M | {s['tflite_bytes'] / 1024:.0f} KB | "
                     f"{s['peak_activation_bytes'] / 1024:.0f} KB | {', '.join(s['ops'])} | {'yes' if s['qat'] else 'no'} | "
                     f"{s['pc_tflite_latency']['median_ms']:.2f} ms |")
        c = load(run / "metrics_capsule.json")
        if c:
            for k in ("float", "int8"):
                r, cm = c[k], c[k]["confusion_matrix"]
                capsule.append(f"| {k.upper() if k == 'int8' else 'Float'} | {size}×{size} | {f(r['recall'])} | "
                               f"{f(r['specificity'])} | {f(r['roc_auc'])} | {cm['tn']} / {cm['fp']} / {cm['fn']} / {cm['tp']} |")
        shutil.copy(run / "metrics_test.json", OUT / f"classifier_{size}_metrics_test.json")
        if c:
            shutil.copy(run / "metrics_capsule.json", OUT / f"classifier_{size}_metrics_capsule.json")
    lines += ["", "| Input | Params | MACs | .tflite | Peak activations | Ops | QAT | PC interpreter (host, 1 thread)* |",
              "|---|---|---|---|---|---|---|---|", *stats, "",
              "\\* TFLite built-in optimised kernels (no XNNPACK) on the PC: a sanity check, **not** board latency. "
              "The board must fit the model in 512 KB RAM; peak activations are well inside it."]
    if capsule:
        lines += ["", "### Kvasir-Capsule only (domain shift)", "",
                  "Kvasir-Capsule polyp frames (all from one video) + normal-mucosa frames from test videos. "
                  "Same threshold as above. Never used for training.", "",
                  "| Model | Input | Recall | Specificity | ROC-AUC | TN / FP / FN / TP |", "|---|---|---|---|---|---|", *capsule]
    return lines


BOARD_MODES = {"pipeline_board": "JPEG (FRM0): 336×336 JPEG, decoded on the board at 1/4 scale",
               "pipeline_board_raw": "raw (RAW0): 84×84 RGB, area-resized on the PC, no decode on the board"}


def auc(y, s):
    """ROC-AUC as the Mann-Whitney U statistic (ties count half)."""
    y, s = np.asarray(y), np.asarray(s, float)
    order = s.argsort()
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):  # average ranks of ties
        ranks[s == v] = ranks[s == v].mean()
    pos = y == 1
    return (ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())


def board_section(runs, pc_csv):
    """Classifier on the nRF54LM20 DK: test-set composition, metrics and per-frame times for each board run."""
    runs = [r for r in runs if (r / "board.csv").exists()]
    if not runs:
        return []
    labels = list(csv.DictReader(open(ROOT / "Data/test/labels.csv")))
    y_of = {int(r["frame_id"]): int(r["polyp"]) for r in labels}
    groups = {}
    for r in labels:
        k = (r["source"].rsplit("/", 1)[0], "polyp" if r["polyp"] == "1" else "no polyp")
        groups[k] = groups.get(k, 0) + 1
    n_pos = sum(y_of.values())
    lines = ["## Classifier on the nRF54LM20 DK (measured on the board)", "",
             "MobileNetV1 0.25 INT8, 84×84 input (`runs/classifier_84`), compiled with the Axon NN compiler and run on "
             "the Axon NPU of the nRF54LM20B; threshold int8 ≥ 30. Frames are sent one at a time from the PC over USB "
             "CDC-ACM (high speed); see `Board/README.md` for the protocol and how each time is measured.", "",
             f"### Test set: {len(labels)} replay frames ({n_pos} polyp, {len(labels) - n_pos} no polyp)", "",
             "The test split of `Data/splits.json` as 336×336 JPEG frames (`Data/test/frames`, labels in "
             "`Data/test/labels.csv`). None of these images, or copies of them, are used for training.", "",
             "| Source folder | Class | Frames |", "|---|---|---|"]
    lines += [f"| `{src}` | {cls} | {n} |" for (src, cls), n in sorted(groups.items(), key=lambda kv: (kv[0][1], -kv[1]))]
    lines += ["", "`kvasir-dataset/polyps` are the Kvasir v1 copies of the Kvasir-SEG test images (uncropped); "
              "`negatives/kvasir-capsule` are Kvasir-Capsule normal-mucosa frames from the test videos.", ""]

    pc = {int(r["frame_id"]): int(r["detected"]) for r in csv.DictReader(open(pc_csv))} if pc_csv.exists() else {}
    metric_rows, time_rows, total_rows = [], [], []
    for run in runs:
        rows = list(csv.DictReader(open(run / "board.csv")))
        mode = BOARD_MODES.get(run.name, run.name)
        fid = [int(r["frame_id"]) for r in rows]
        y = np.array([y_of[i] for i in fid])
        det = np.array([int(r["detected"]) for r in rows])
        tp, fp = int(((det == 1) & (y == 1)).sum()), int(((det == 1) & (y == 0)).sum())
        fn, tn = int(((det == 0) & (y == 1)).sum()), int(((det == 0) & (y == 0)).sum())
        rec, prec, spec = tp / (tp + fn), tp / max(tp + fp, 1), tn / (tn + fp)
        agree = f"{sum(pc[i] == d for i, d in zip(fid, det))} / {len(fid)}" if pc else "–"
        metric_rows.append(f"| {mode} | {len(rows)} | {f(prec)} | {f(rec)} | {f(2 * prec * rec / (prec + rec))} | "
                           f"{f(spec)} | {f(auc(y, [int(r['raw']) for r in rows]))} | {tn} / {fp} / {fn} / {tp} | {agree} |")
        ns = lambda k: np.array([int(r[k]) for r in rows], np.int64)
        st = {"PC → board (USB, frame payload)": ns("usb_down_ns"),
              "board preprocessing": ns("t_infer_start_ns") - ns("t_rx_board_ns"),
              "**Axon inference**": ns("t_infer_end_ns") - ns("t_infer_start_ns"),
              "board → PC (result line + frame if detected)": ns("usb_ns")}
        st["**board compute (preprocessing + inference)**"] = st["board preprocessing"] + st["**Axon inference**"]
        st["**round trip per frame (PC send → reply in)**"] = (ns("usb_down_ns") + ns("t_sent_ns") - ns("t_rx_board_ns")
                                                              + ns("usb_ns"))
        for name, v in st.items():
            ms = v / 1e6
            time_rows.append(f"| {mode.split(':')[0]} | {name} | {np.median(ms):.2f} | {np.percentile(ms, 95):.2f} | "
                             f"{ms.mean():.2f} | {ms.min():.2f} | {ms.max():.2f} | {ms.sum() / 1000:.2f} s |")
        tag = {"pipeline_board": "jpeg", "pipeline_board_raw": "raw"}.get(run.name, run.name)
        shutil.copy(run / "board.csv", OUT / f"board_{tag}_frames.csv")
    lines += ["### Metrics on the board", "",
              "| Input sent to the board | Frames | Precision | Recall | F1 | Specificity | ROC-AUC (raw int8) | "
              "TN / FP / FN / TP | Same decision as the PC INT8 model |", "|---|---|---|---|---|---|---|---|---|",
              *metric_rows, "",
              "### Time per frame and in total (board clock for the board stages, host clock for board → PC)", "",
              "| Input | Stage | p50 ms | p95 ms | mean ms | min ms | max ms | total, all frames |",
              "|---|---|---|---|---|---|---|---|", *time_rows, "",
              "Per-frame values: `board_jpeg_frames.csv`, `board_raw_frames.csv` (columns described in "
              "`Board/receiver.py`). In raw mode the PC's decode + resize to 84×84 is not included; on the pill that "
              "work moves to the sender."]
    return lines


def yolo_section(runs):
    lines = ["## YOLO26n box detector (PC)", "",
             "`test` = Kvasir-SEG test (100 images) + every test negative as a background image (648), so false "
             "positives count. `test_pos` = the 100 polyp images only. mAP at conf 0.001; tightness and false alarms "
             "at the deployment confidence (F1-optimal on val).", "",
             "| Model | Format | Split | Precision | Recall | mAP50 | mAP50-95 |", "|---|---|---|---|---|---|---|"]
    tight = []
    for run in runs:
        m = load(run / "metrics_eval.json")
        if not m:
            continue
        shutil.copy(run / "metrics_eval.json", OUT / f"{run.name}_metrics_eval.json")
        for fmt, r in m["formats"].items():
            for split in ("test", "test_pos"):
                v = r[split]
                lines.append(f"| {run.name} | {fmt} | {split} | {f(v['precision'])} | {f(v['recall'])} | "
                             f"{f(v['mAP50'])} | {f(v['mAP50-95'])} |")
            t, fa = r["tightness"], r["false_alarms"]
            tight.append(f"| {run.name} | {fmt} | {m['deploy_conf']:.3f} | {t['matched']}/{t['gt_boxes']} | "
                         f"{f(t['mean_iou'])} | {f(t['median_iou'])} | {t['abs_w_err_px336']:.1f} / {t['abs_h_err_px336']:.1f} | "
                         f"{t['signed_w_err_px336']:+.1f} / {t['signed_h_err_px336']:+.1f} | "
                         f"{fa['with_box']}/{fa['background_images']} ({fa['false_alarm_rate']:.1%}) |")
    lines += ["", "### Box tightness (test_pos) and false alarms (test backgrounds)", "",
              "Greedy one-to-one matching at IoU ≥ 0.5. Width/height errors in pixels of the 336×336 frame; "
              "signed = predicted minus ground truth (positive = box too large).", "",
              "| Model | Format | Conf | Matched | Mean IoU | Median IoU | abs W / H err (px) | signed W / H err (px) | "
              "Backgrounds with a box |", "|---|---|---|---|---|---|---|---|---|", *tight]
    return lines


def bench_section():
    lines = []
    for path in sorted(OUT.glob("yolo_bench_*.json")):
        lines += bench_table(load(path), path.stem.removeprefix("yolo_bench_")) + [""]
    return lines


def bench_table(b, run):
    lines = [f"### {run}: CPU latency on this PC (measured)", "",
             f"{b['machine']['cpu']}, {b['machine']['platform']}; batch 1, {b['frame']} frames, {b['warmup']} warm-up + "
             f"{b['runs']} timed frames per row, one model at a time on an idle machine. p50 / p95 in ms.", "",
             "| Format | Threads | Pre | Infer | Post | **Total** |", "|---|---|---|---|---|---|"]
    for r in b["results"]:
        c = lambda s: f"{r[s]['p50_ms']:.2f} / {r[s]['p95_ms']:.2f}"
        lines.append(f"| {r['format']} | {r['threads']} | {c('pre')} | {c('infer')} | {c('post')} | **{c('total')}** |")
    return lines


def latency_section(pipe):
    s = load(pipe / "latency_summary.json")
    if not s:
        return []
    for name in ("latency_breakdown.png", "latency_breakdown.pdf", "latency_stages.csv", "latency_summary.json"):
        if (pipe / name).exists():
            shutil.copy(pipe / name, OUT / name)
    rows = [("1 · frame → board", "1_frame_to_board"), ("2a · board preprocessing", "2a_board_preprocess"),
            ("2b · board inference", "2b_board_inference"), ("3 · board → PC (USB)", "3_board_to_pc"),
            ("4 · YOLO26n on PC CPU", "4_yolo_pc_cpu"), ("&nbsp;&nbsp;4a · pre", "4a_yolo_pre"),
            ("&nbsp;&nbsp;4b · infer", "4b_yolo_infer"), ("&nbsp;&nbsp;4c · post", "4c_yolo_post"),
            ("**Detection total** (2 + 3 + 4)", "detection_total"), ("5 · simulation per box", "5_simulation_per_box")]
    lines = ["## Latency per stage (pipeline run)", "",
             "| Stage | p50 ms | p95 ms | n | Label | How / where |", "|---|---|---|---|---|---|"]
    for title, k in rows:
        v = s[k]
        lines.append(f"| {title} | {f(v['p50_ms'], 2)} | {f(v['p95_ms'], 2)} | {v['n']} | {v.get('label', 'measured')} | "
                     f"{v.get('how', '')} |")
    d = s["detection_total"]
    verdict = "**within**" if d["within_budget"] else "**over**"
    lines += ["", f"Detection p95 = **{f(d['p95_ms'], 1)} ms** vs the 100 ms target: {verdict} budget "
              f"({d['label']}; stage 1 is not included). Counts: {s['counts']}.", "",
              "![latency breakdown](latency_breakdown.png)", "",
              "In the figure, *detection* is the measured value from this run; *capture* and *network* are still the "
              "visualizer's fake values (hatched); *lookup* is the full PBS run per box until the lookup table exists."]
    e = s.get("end_to_end")
    if e:
        lines += ["", "### End to end (classifier + YOLO26n on the replay frames)", "", "| | |", "|---|---|",
                  f"| Kvasir-SEG polyp frames flagged by the classifier | {e['gt_box_frames_flagged']} / {e['gt_box_frames']} |",
                  f"| Kvasir-SEG polyp frames ending with a correct box (IoU ≥ 0.5) | **{e['gt_box_frames_correct_box']} / {e['gt_box_frames']}** |",
                  f"| All polyp frames ending with any box | {e['polyp_frames_with_box']} / {e['polyp_frames']} |",
                  f"| Polyp-free frames flagged by the classifier | {e['negative_frames_flagged']} / {e['negative_frames']} |",
                  f"| Polyp-free frames ending with a box (false injection) | **{e['negative_frames_with_box']} / {e['negative_frames']}** |"]
    return lines


def examples(pipe, n=6):
    """Frames with predicted (orange) vs ground-truth (green) boxes."""
    if not (pipe / "boxes.csv").exists():
        return []
    gt, pred = {}, {}
    for r in csv.DictReader(open(ROOT / "Data/test/gt_boxes.csv")):
        gt.setdefault(int(r["frame_id"]), []).append([float(r[k]) for k in ("x_min", "y_min", "x_max", "y_max")])
    for r in csv.DictReader(open(pipe / "boxes.csv")):
        pred.setdefault(int(r["frame_id"]), []).append(
            [float(r[k]) for k in ("x_min", "y_min", "x_max", "y_max", "confidence")])
    ids = sorted(set(gt) & set(pred))[:n]
    if not ids:
        return []
    tiles = []
    for fid in ids:
        img = cv2.imread(str(ROOT / f"Data/test/frames/frame_{fid:06d}.jpg"))
        for b in gt[fid]:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (60, 200, 60), 2)
        for b in pred[fid]:
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (20, 120, 240), 2)
            cv2.putText(img, f"{b[4]:.2f}", (int(b[0]) + 3, max(int(b[1]) - 4, 12)), 0, 0.5, (20, 120, 240), 2)
        cv2.putText(img, f"frame {fid}", (6, 330), 0, 0.5, (255, 255, 255), 1)
        tiles.append(img)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(OUT / "examples.jpg"), np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
    return ["### Example frames", "", "Green = ground truth (Kvasir-SEG), orange = YOLO26n with confidence.", "",
            "![examples](examples.jpg)"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pipeline", type=Path, default=ROOT / "runs/pipeline")
    p.add_argument("--yolo", type=Path, nargs="+", default=[ROOT / "runs/yolo26n", ROOT / "runs/yolo26n_p2"])
    p.add_argument("--classifier", type=Path, nargs="+", default=[ROOT / "runs/classifier_96", ROOT / "runs/classifier_84"])
    p.add_argument("--board", type=Path, nargs="+", default=[ROOT / "runs/pipeline_board", ROOT / "runs/pipeline_board_raw"],
                   help="board runs with board.csv from Board/receiver.py (skipped if missing)")
    p.add_argument("--open", type=Path, default=ROOT / "results/open_items.md", help="hand-written 'still open' list")
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    doc = ["# Detection results", "",
           "All numbers on the held-out test split shared by both models (`Data/splits.json`); no test image, and no "
           "copy of one, is used for training. Generated by `report.py` from the JSON/CSV files in this folder.", ""]
    doc += classifier_section(a.classifier) + [""] + board_section(a.board, ROOT / "runs/pipeline/classifier.csv") + [""]
    doc += yolo_section(a.yolo) + [""] + bench_section() + [""]
    doc += latency_section(a.pipeline) + [""] + examples(a.pipeline) + [""]
    if a.open.exists():
        doc += [a.open.read_text()]
    (OUT / "RESULTS.md").write_text("\n".join(doc))
    print(f"[report] -> {OUT / 'RESULTS.md'}")


if __name__ == "__main__":
    main()

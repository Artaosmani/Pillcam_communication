"""Demo overlay (optional): video frame with its box, predicted Pi_T and the
inject decision drawn on top. One PNG per box, offline only.

Inputs
  --boxes    detector log (boxes_fake.csv layout)
  --server   bbox_replay_out.csv
  --frames   folder with frame images named <frame_id>.png / .jpg (optional).
             Without it, or when a frame is missing, a grey PLACEHOLDER frame
             of size img_w x img_h is drawn and labelled as such.
Outputs
  overlay_<frame_id>_t<track_id>.png, and overlay_contact_sheet.png with all of them

Run
  python demo_overlay.py --boxes ../PBS_package/drift/data/boxes_fake.csv \
      --server ../PBS_package/drift/bbox_replay_out.csv --out-dir out
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import vizstyle as vs


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_frame(frames_dir, frame_id, w, h):
    if frames_dir:
        for ext in (".png", ".jpg", ".jpeg"):
            p = os.path.join(frames_dir, f"{frame_id}{ext}")
            if os.path.exists(p):
                return plt.imread(p), False
    return np.full((h, w, 3), 0.42), True


def draw(ax, b, s, frames_dir):
    w, h = int(float(b["img_w"])), int(float(b["img_h"]))
    img, placeholder = load_frame(frames_dir, b["frame_id"], w, h)
    ax.imshow(img, extent=(0, w, h, 0))
    if placeholder:
        ax.text(w - 4, 4, "PLACEHOLDER (no video frame)", ha="right", va="top",
                color="white", alpha=0.7, fontsize=7)
    inject = int(s["inject"]) == 1
    color = vs.GREEN if inject else vs.RED
    x0, y0, x1, y1 = (float(b[k]) for k in ("x_min", "y_min", "x_max", "y_max"))
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=2.5))
    ax.text(x0, y0 - 4, f"polyp {float(b['confidence']):.2f}", color="white", fontsize=8,
            va="bottom", bbox=dict(fc=color, ec="none", pad=1.5))
    decision = f"INJECT  N = {float(s['dose_N']):.0f}" if inject else "NO INJECT"
    lines = [decision,
             f"Pi_T = {float(s['Pi_T']):.3f}   t½ = {float(s['t_half_s']) * 1e3:.3f} ms",
             f"server latency {float(s['latency_ms']):.1f} ms"]
    ax.text(6, h - 6, "\n".join(lines), color="white", fontsize=8, va="bottom", family="monospace",
            bbox=dict(fc="black", alpha=0.6, ec="none", pad=4))
    ax.set_title(f"frame {b['frame_id']}  track {b['track_id']}", loc="left", fontsize=9)
    ax.set_xlim(0, w); ax.set_ylim(h, 0)
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--frames", default="")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    vs.apply()

    bmap = {(r["frame_id"], r["track_id"]): r for r in read_csv(a.boxes)}
    pairs = [(bmap[(s["frame_id"], s["track_id"])], s) for s in read_csv(a.server)
             if (s["frame_id"], s["track_id"]) in bmap]

    for b, s in pairs:
        fig, ax = plt.subplots(figsize=(4, 4))
        draw(ax, b, s, a.frames)
        fig.savefig(os.path.join(a.out_dir, f"overlay_{b['frame_id']}_t{b['track_id']}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    cols = 4
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (b, s) in zip(axes.flat, pairs):
        draw(ax, b, s, a.frames)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "overlay_contact_sheet.png"), dpi=150, bbox_inches="tight")
    print(f"{len(pairs)} overlays -> {a.out_dir}")


if __name__ == "__main__":
    main()

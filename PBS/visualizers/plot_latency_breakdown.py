"""Latency breakdown per frame: one stacked bar per detected box.

Stages (pipeline order): capture, detection, network, mapping, lookup,
decision, plus the physical delivery time t_half_s as a separate segment
on top. Built offline from the two CSV logs, never inside the timed loop.

Inputs
  --boxes   detector log, one row per box (same layout as boxes_fake.csv):
            frame_id,track_id,t_capture_ns,t_detect_ns,t_send_ns,x_min,...
            Optional extra column capture_ms (sensor/readout time before
            t_capture_ns) is used when present.
  --server  server log, bbox_replay_out.csv from bbox_replay.cpp:
            ...,latency_ms,mapping_ms,lookup_ms,decision_ms
            Optional column t_recv_ns (live server) gives network = t_recv - t_send.

Where a stage has no real measurement yet, a FAKE value is generated
(constants below), the bar segment is hatched, and the legend and the
merged CSV say so. Replace the inputs and the fakes disappear by themselves.
detection is marked fake when the boxes file name contains "fake"
(override with --real-boxes).

Outputs
  latency_breakdown.png / .pdf   the figure
  latency_stages.csv             merged per-box stages in ms, with fake_stages

Run
  python plot_latency_breakdown.py --boxes ../PBS_package/drift/data/boxes_fake.csv \
      --server ../PBS_package/drift/bbox_replay_out.csv --out-dir out
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import vizstyle as vs

# FAKE stage models (mean, std) in ms, used only while no real log exists.
# capture: capsule sensor + readout, network: gateway -> server box message.
FAKE = {"capture": (8.0, 1.5), "network": (3.0, 1.0)}
BUDGET_MS = 100.0   # p95 per-frame target (background.md, "Latency target")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_stages(boxes, server, boxes_fake, seed=0):
    rng = np.random.default_rng(seed)
    key = lambda r: (r["frame_id"], r["track_id"])
    bmap = {key(r): r for r in boxes}
    rows = []
    for s in server:
        b = bmap.get(key(s))
        if b is None:
            print(f"warning: no detector row for frame {s['frame_id']} track {s['track_id']}")
            continue
        fake = []
        r = {"frame_id": s["frame_id"], "track_id": s["track_id"]}

        if b.get("capture_ms"):
            r["capture"] = float(b["capture_ms"])
        else:
            r["capture"] = max(0.0, rng.normal(*FAKE["capture"]))
            fake.append("capture")

        r["detection"] = (int(b["t_detect_ns"]) - int(b["t_capture_ns"])) / 1e6
        if boxes_fake:
            fake.append("detection")

        if s.get("t_recv_ns"):
            r["network"] = (int(s["t_recv_ns"]) - int(b["t_send_ns"])) / 1e6
        else:
            r["network"] = max(0.0, rng.normal(*FAKE["network"]))
            fake.append("network")

        r["mapping"] = float(s["mapping_ms"])
        r["lookup"] = float(s["lookup_ms"])
        r["decision"] = float(s["decision_ms"])
        r["t_half_ms"] = float(s["t_half_s"]) * 1e3
        r["inject"] = int(s["inject"])
        r["fake_stages"] = ";".join(fake)
        rows.append(r)
    return rows


def write_stages(rows, path):
    cols = ["frame_id", "track_id"] + vs.STAGES + ["t_half_ms", "inject", "fake_stages"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.4f}" if isinstance(r[c], float) else r[c]) for c in cols})


def plot(rows, out_base, title_note="", subtitle="", max_bars=0):
    vs.apply()
    all_rows = rows
    if max_bars and len(rows) > max_bars:  # panel A: first boxes only; panel B uses every box
        rows = rows[:max_bars]
        title_note += f" (first {max_bars} of {len(all_rows)})"
    n = len(rows)
    x = np.arange(n)
    labels = [f"{r['frame_id']}\nt{r['track_id']}" for r in rows]
    any_fake = {st for r in rows for st in r["fake_stages"].split(";") if st}

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [2.3, 1]})

    # Panel A: stacked bar per box
    bottom = np.zeros(n)
    width = 0.62
    for st in vs.STAGES + ["t_half_ms"]:
        vals = np.array([r[st] for r in rows])
        color = vs.PHYS_COLOR if st == "t_half_ms" else vs.STAGE_COLORS[st]
        hatches = ["///" if st in r["fake_stages"].split(";") else "" for r in rows]
        for i in range(n):
            if vals[i] <= 0:
                continue
            ax.bar(x[i], vals[i], width, bottom=bottom[i], color=color,
                   hatch=hatches[i], edgecolor=vs.SURFACE, linewidth=1.0)
        bottom += vals
    totals = bottom
    for i, r in enumerate(rows):
        ax.annotate(f"{totals[i]:.0f}", (x[i], totals[i]), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8, color=vs.TEXT2)
        if not r["inject"]:
            ax.annotate("no inject", (x[i], totals[i]), xytext=(0, 14),
                        textcoords="offset points", ha="center", fontsize=8, color=vs.TEXT2)
    ax.axhline(BUDGET_MS, color=vs.TEXT, lw=1, ls="--")
    ax.annotate(f"{BUDGET_MS:.0f} ms p95 budget", (-0.5, BUDGET_MS), xytext=(2, 4),
                textcoords="offset points", ha="left", fontsize=8, color=vs.TEXT,
                bbox=dict(boxstyle="square,pad=0.1", fc=vs.SURFACE, ec="none"))
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_xlabel("frame / track")
    ax.set_ylabel("time per box [ms]")
    ax.set_ylim(0, max(totals.max(), BUDGET_MS) * 1.12)
    ax.grid(axis="x", visible=False)
    ax.set_title("Where the time goes, per detected box" + title_note, loc="left", pad=18)
    if subtitle:
        ax.annotate(subtitle, (0, 1.01), xycoords="axes fraction", fontsize=8, color=vs.TEXT2)

    handles = [Patch(color=vs.STAGE_COLORS[s], label=s + (" (fake)" if s in any_fake else ""))
               for s in vs.STAGES]
    handles.append(Patch(color=vs.PHYS_COLOR, label="physical delivery t½"))
    if any_fake:
        handles.append(Patch(facecolor=vs.MUTED, hatch="///", edgecolor=vs.SURFACE,
                             label="hatched = fake data"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.13), ncol=4, fontsize=8)

    # Panel B: mean and p95 per stage (dot plot, log x so tiny stages stay visible)
    names = vs.STAGES + ["t_half_ms"]
    y = np.arange(len(names))[::-1]
    for yi, st in zip(y, names):
        v = np.array([r[st] for r in all_rows if (st not in ("lookup", "decision", "t_half_ms") or r["inject"])])
        v = v[v > 0]
        color = vs.PHYS_COLOR if st == "t_half_ms" else vs.STAGE_COLORS[st]
        if len(v) == 0:
            continue
        mean, p95 = v.mean(), np.percentile(v, 95)
        ax2.plot([v.min(), v.max()], [yi, yi], color=color, lw=2, solid_capstyle="round", alpha=0.45)
        ax2.plot(mean, yi, "o", ms=8, color=color, mec=vs.SURFACE, mew=2)
        ax2.plot(p95, yi, "|", ms=12, mew=2, color=vs.TEXT)
        ax2.annotate(f"{mean:.3g} ms", (mean, yi), xytext=(0, 8), textcoords="offset points",
                     ha="center", fontsize=8, color=vs.TEXT2)
    ax2.set_xscale("log")
    ax2.set_yticks(y, [("t½ (physical)" if s == "t_half_ms" else s) +
                       (" *" if s in any_fake else "") for s in names])
    ax2.set_xlabel("ms (log scale)")
    ax2.grid(axis="y", visible=False)
    ax2.set_title("Per stage: mean ●, p95 |, range ─", loc="left")
    if any_fake:
        ax2.annotate("* fake data", (0, -0.13), xycoords="axes fraction", fontsize=8, color=vs.TEXT2)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boxes", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--real-boxes", action="store_true", help="treat detection times as real")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subtitle", default="lookup = full PBS run per box until the lookup table exists")
    ap.add_argument("--max-bars", type=int, default=0, help="stacked bars for the first N boxes only (0 = all)")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    boxes_fake = (not a.real_boxes) and "fake" in os.path.basename(a.boxes).lower()
    rows = build_stages(read_csv(a.boxes), read_csv(a.server), boxes_fake, a.seed)
    write_stages(rows, os.path.join(a.out_dir, "latency_stages.csv"))
    plot(rows, os.path.join(a.out_dir, "latency_breakdown"), subtitle=a.subtitle, max_bars=a.max_bars)
    print(f"{len(rows)} boxes -> {a.out_dir}/latency_breakdown.png, latency_stages.csv")


if __name__ == "__main__":
    main()

"""Particle animation from particle_snapshots.csv (offline, for talks and debugging).

2-D projection: the transmitter is at the origin and the receiver centre at
(r0, 0, 0), so the setup is symmetric around the x axis. Each particle is
drawn at (x, ±sqrt(y² + z²)), with a fixed random sign per particle. This
keeps the distance to the receiver centre exact: a dot is inside the circle
if and only if the particle is inside the sphere, so a FREE (blue) dot
inside the circle would mean the push-back after a failed binding is broken.

Panels: full view, zoom on the receiver, and state counts over time.

Inputs   particle_snapshots.csv and particle_snapshots_meta.csv
         (from PBS_package/drift/particle_snapshots.cpp)
Outputs  particles.mp4 (or .gif with --gif) and particles_frames.png
         (four stills for the thesis / slides)

Run
  python animate_particles.py --csv ../PBS_package/drift/particle_snapshots.csv --out-dir out
"""
import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D

import vizstyle as vs

UM = 1e6


def load(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    meta_path = path[: path.rfind(".")] + "_meta.csv"
    with open(meta_path, newline="") as f:
        meta = {k: float(v) for k, v in next(csv.DictReader(f)).items()}
    steps = np.unique(data["step"]).astype(int)
    frames = []
    rng = np.random.default_rng(1)
    sign = {}
    for s in steps:
        d = data[data["step"] == s]
        ids = d["id"].astype(int)
        for i in ids:
            if i not in sign:
                sign[i] = rng.choice([-1.0, 1.0])
        sg = np.array([sign[i] for i in ids])
        x = d["x_m"] * UM
        rho = np.sqrt(d["y_m"] ** 2 + d["z_m"] ** 2) * UM * sg
        frames.append((s, d["t_s"][0], x, rho, d["state"].astype(int)))
    return frames, meta


def counts(frames, N):
    out = np.zeros((len(frames), 4))
    for k, (_, _, _, _, st) in enumerate(frames):
        for j in range(4):
            out[k, j] = np.sum(st == j)
    return out


def make_figure(frames, meta, full_lim):
    vs.apply()
    r0, a = meta["r0_m"] * UM, meta["a_m"] * UM
    N = int(meta["N"])
    fig = plt.figure(figsize=(12, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1.15])
    axf, axz, axc = (fig.add_subplot(gs[0, i]) for i in range(3))

    for ax, lim, title in ((axf, full_lim, "Full view"),
                           (axz, max(2.2 * a, 1.0), "Zoom on receiver")):
        ax.add_patch(plt.Circle((r0, 0), a, fc="none", ec=vs.TEXT, lw=1.5))
        ax.plot(0, 0, marker="*", ms=12, color=vs.RED, mec=vs.SURFACE, zorder=5)
        cx = r0 if ax is axz else r0 / 2
        ax.set_xlim(cx - lim, cx + lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(title, loc="left")
        ax.set_xlabel("x along Tx→Rx axis [µm]")
    axf.set_ylabel("±√(y²+z²) [µm]")

    scat_f = axf.scatter([], [], s=10, linewidths=0)
    scat_z = axz.scatter([], [], s=22, linewidths=0.6, edgecolors=vs.SURFACE)
    clock = axf.annotate("", (0.02, 0.97), xycoords="axes fraction", va="top", fontsize=9)

    c = counts(frames, N)
    t_ms = np.array([f[1] for f in frames]) * 1e3
    # end labels, spread so they never overlap
    ends = c[-1] / N
    ys = ends.copy()
    order = np.argsort(ys)
    for prev, cur in zip(order[:-1], order[1:]):
        ys[cur] = max(ys[cur], ys[prev] + 0.05)
    for j in range(4):
        axc.plot(t_ms, c[:, j] / N, color=vs.STATE_COLORS[j], lw=2, label=vs.STATE_NAMES[j])
        axc.annotate(vs.STATE_NAMES[j], (t_ms[-1], ends[j]), xytext=(t_ms[-1] * 1.02, ys[j]),
                     textcoords="data", va="center", fontsize=8, color=vs.TEXT2)
    cursor = axc.axvline(0, color=vs.TEXT, lw=1)
    axc.set_xlim(0, t_ms[-1] * 1.18)
    axc.set_ylim(0, 1)
    axc.set_xlabel("t [ms]")
    axc.set_ylabel("fraction of particles")
    axc.set_title("State fractions", loc="left")

    handles = [Line2D([], [], ls="", marker="o", ms=7, color=vs.STATE_COLORS[j],
                      label=vs.STATE_NAMES[j]) for j in range(4)]
    handles += [Line2D([], [], ls="", marker="*", ms=10, color=vs.RED, label="release point (Tx)"),
                Line2D([], [], color=vs.TEXT, lw=1.5, label="receiver sphere")]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    note = (f"N = {N}, r0 = {r0:.2f} µm, a = {a:.2f} µm, dt = {meta['dt']:.0e} s, "
            f"free particles inside sphere after receiver step: {int(meta['free_inside_after_process'])}")
    fig.suptitle(note, fontsize=8, color=vs.TEXT2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))

    colors = np.array(vs.STATE_COLORS)

    def draw(k):
        _, t, x, rho, st = frames[k]
        order = np.argsort(st == 0)   # draw free first, others on top
        xy = np.c_[x, rho][order]
        cols = colors[st[order]]
        scat_f.set_offsets(xy); scat_f.set_color(cols)
        scat_z.set_offsets(xy); scat_z.set_facecolor(cols)
        clock.set_text(f"t = {t * 1e3:.3f} ms")
        cursor.set_xdata([t * 1e3, t * 1e3])
        return scat_f, scat_z, clock, cursor

    return fig, draw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--gif", action="store_true", help="write GIF instead of MP4")
    ap.add_argument("--lim", type=float, default=0, help="half-width of full view [µm], 0 = auto")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    frames, meta = load(a.csv)
    lim = a.lim
    if lim <= 0:
        allx = np.concatenate([np.abs(f[2] - meta["r0_m"] * UM / 2) for f in frames])
        allr = np.concatenate([np.abs(f[3]) for f in frames])
        lim = float(np.percentile(np.r_[allx, allr], 98)) * 1.05

    fig, draw = make_figure(frames, meta, lim)

    # four stills
    still_idx = np.linspace(0, len(frames) - 1, 4).round().astype(int)
    for i, k in enumerate(still_idx):
        draw(k)
        fig.savefig(os.path.join(a.out_dir, f"particles_t{i}.png"), dpi=160, bbox_inches="tight")
    ims = [plt.imread(os.path.join(a.out_dir, f"particles_t{i}.png")) for i in range(4)]
    h = min(im.shape[0] for im in ims); w = min(im.shape[1] for im in ims)
    grid = np.vstack([im[:h, :w] for im in ims])
    plt.imsave(os.path.join(a.out_dir, "particles_frames.png"), grid)
    for i in range(4):
        os.remove(os.path.join(a.out_dir, f"particles_t{i}.png"))

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), blit=False)
    if a.gif:
        out = os.path.join(a.out_dir, "particles.gif")
        anim.save(out, writer=animation.PillowWriter(fps=a.fps), dpi=90)
    else:
        out = os.path.join(a.out_dir, "particles.mp4")
        anim.save(out, writer=animation.FFMpegWriter(fps=a.fps, bitrate=2400), dpi=120)
    print(f"{len(frames)} frames -> {out}, particles_frames.png")


if __name__ == "__main__":
    main()

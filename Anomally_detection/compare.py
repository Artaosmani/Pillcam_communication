"""Edge vs board: uplink bandwidth, end-to-end latency with the injection command back to the board, accuracy and
energy per frame, for three ways to split the work. Writes results/COMPARISON.md + results/comparison.json.

    A  edge only    board forwards every frame, YOLO26n on the edge            receiver.py --fwd --edge-model ...
    B  gated        classifier on the board (Axon), positives to the edge     receiver.py --raw --edge-model ...
    C  board boxes  YOLO26 (Axon variant) on the board, only boxes go up     receiver.py --det --conf <deploy>
    C+ = C gated by B's classifier (detector only on classifier positives), combined per frame from the B and C runs

python compare.py --fwd runs/compare/fwd.csv --gated runs/compare/gated.csv --det runs/compare/det.csv

Measured (from the CSVs): board compute, USB transfer, edge YOLO26n time, closed-loop e2e (board clock: frame in
-> command in), boxes. Modelled: the BLE uplink (throughput and connection interval below) and all energy numbers
(nRF54LM20 datasheet v1.0 currents x measured durations; no current was measured, there is no power meter).
The uplink payload of A and B is counted as the frame's JPEG (what the pill sends and the edge's YOLO26n reads).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "YOLO"))
from evaluate import match  # noqa: E402

ROOT = Path(__file__).resolve().parent
V = 3.0  # datasheet currents are at 3.0 V
# nRF54LM20A/B datasheet v1.0, section 11.2 (typ., 3 V); the DK's other parts (debugger, LEDs) are not counted
I_MA = {"cpu": 2.4,      # IAPPCPU0, CPU running CoreMark at 128 MHz from NVM, cache on
        "axon": 2.8,     # IAXONS2, Axon NPU running MobileNetV1 (VWW)
        "tx": 5.0,       # IRADIO_TX0, radio TX 0 dBm
        "rx": 3.3,       # IRADIO_RX0, radio RX 1 Mbps
        "idle": 0.0043}  # IONIDLE2, System ON, GRTC, 512 KB RAM retained
LINE_B, CMD_B = 70, 20  # result line without boxes, command message (CMD0 header + 8 bytes)


def rows(path):
    return {int(r["frame_id"]): r for r in csv.DictReader(open(path))} if path else {}


def f(r, k):
    return float(r[k]) if r.get(k) not in (None, "") else None


def pct(v):
    v = [x for x in v if x is not None]
    return {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "n": len(v)} if v else None


def accuracy(boxes, labels, gt):
    """boxes: {frame_id: [[x0,y0,x1,y1,conf], ...]} in 336 px. Same end-to-end counts as report.py."""
    seg = [i for i in gt]
    return {"seg_correct_box": sum(bool(match(sorted(boxes.get(i, []), key=lambda b: -b[4]), np.array(gt[i])))
                                   for i in seg), "seg_frames": len(seg),
            "polyp_any_box": sum(bool(boxes.get(i)) for i, p in labels.items() if p),
            "polyp_frames": sum(labels.values()),
            "false_injection": sum(bool(boxes.get(i)) for i, p in labels.items() if not p),
            "negative_frames": sum(1 for p in labels.values() if not p)}


def ble(bytes_, a):
    """Modelled BLE time (ms) for one message: half a connection interval of waiting + airtime."""
    return a.ble_ci_ms / 2 + bytes_ * 8 / a.ble_kbps


def scenario(name, frames, jpeg, a, det_ms=None, gate=None):
    """Per-frame bytes, times and energy for one scenario. frames: {id: receiver row}; gate: B's rows for C+."""
    out = {"frames": {}, "name": name}
    for i, r in frames.items():
        on = gate is None or int(gate[i]["detected"])
        boxes = json.loads(r["boxes"]) if r.get("boxes") and on else []
        fwd = name in ("A", "B") and int(r["detected"])
        up = LINE_B + jpeg[i] if fwd else LINE_B + 26 * len(boxes)  # A/B: the pill's JPEG goes up
        if name == "C+":
            clf_ms, clf_inf = f(gate[i], "board_ms"), (int(gate[i]["t_infer_end_ns"]) - int(gate[i]["t_infer_start_ns"])) / 1e6
            board, axon = clf_ms + (f(r, "board_ms") if on else 0), clf_inf + (det_ms[i] if on else 0)
        else:
            board = f(r, "board_ms")
            axon = (int(r["t_infer_end_ns"]) - int(r["t_infer_start_ns"])) / 1e6
        edge = f(r, "edge_ms") if name in ("A", "B") else 0.0
        has_cmd = bool(boxes)
        # energy (mJ) on the board per frame: CPU while it computes outside Axon, Axon, radio TX for the uplink,
        # radio RX for the command; the frame's own reception from the pill is the same in every scenario
        tx_ms = up * 8 / a.ble_kbps
        e = {"cpu": I_MA["cpu"] * (board - axon), "axon": I_MA["axon"] * axon, "tx": I_MA["tx"] * tx_ms,
             "rx_cmd": I_MA["rx"] * (CMD_B * 8 / a.ble_kbps) if has_cmd else 0.0}
        e = {k: v * V / 1000 for k, v in e.items()}  # mA*ms*V = uJ -> mJ
        usb_e2e = f(r, "e2e_ms") if name != "C+" else (f(r, "e2e_ms") + f(gate[i], "board_ms") if has_cmd else None)
        ble_e2e = board + ble(up, a) + edge + ble(CMD_B, a) if has_cmd else None
        out["frames"][i] = {"up": up, "board": board, "edge": edge, "boxes": boxes, "usb_e2e": usb_e2e,
                            "ble_e2e": ble_e2e, "e": e, "active_ms": board + tx_ms}
    return out


def summary(s, labels, gt, a):
    fr = s["frames"].values()
    n = len(fr)
    up = sum(x["up"] for x in fr) / n
    e = {k: sum(x["e"][k] for x in fr) / n for k in ("cpu", "axon", "tx", "rx_cmd")}
    e_frame = sum(e.values())
    active = sum(x["active_ms"] for x in fr) / n
    batt = {}
    for fps in a.fps:
        i_avg = e_frame / V * fps + I_MA["idle"] * max(0.0, 1 - active * fps / 1000)  # mJ/s / V = mA
        batt[fps] = {"avg_mA": i_avg, "hours": a.battery_mah / i_avg,
                     "uplink_kbps": up * 8 * fps / 1000, "ble_load": up * 8 * fps / 1000 / a.ble_kbps}
    return {"uplink_B_per_frame": up, "frames_sent_up": sum(x["up"] > LINE_B + 26 * 5 for x in fr),
            "board_ms": pct([x["board"] for x in fr]), "edge_ms": pct([x["edge"] for x in fr if x["boxes"]]),
            "usb_e2e_ms": pct([x["usb_e2e"] for x in fr]), "ble_e2e_ms": pct([x["ble_e2e"] for x in fr]),
            "energy_mJ_per_frame": {**e, "total": e_frame}, "battery": batt,
            "accuracy": accuracy({i: x["boxes"] for i, x in s["frames"].items()}, labels, gt)}


def write_md(res, a, path):
    names = {"A": "A · edge only (all frames up, YOLO26n on edge)", "B": "B · gated (classifier on board)",
             "C": "C · boxes on board (YOLO26-Axon)", "C+": "C+ · classifier → YOLO26-Axon on board"}
    L = ["# Edge vs board: bandwidth, closed-loop latency, accuracy, energy", "",
         "Test set: the 801 replay frames (153 polyp, 648 none). Generated by `compare.py` from the receiver CSVs "
         "(`Board/receiver.py`). **Measured**: board compute, USB, edge YOLO26n, closed-loop e2e on USB, boxes. "
         f"**Modelled**: BLE uplink ({a.ble_kbps:.0f} kbit/s application throughput, {a.ble_ci_ms} ms connection "
         "interval) and all energy (nRF54LM20 datasheet v1.0 currents × measured durations, chip only, 3.0 V; "
         "no current was measured).", "",
         "## Accuracy (end to end)", "",
         "| Scenario | Kvasir-SEG polyp frames with a correct box (IoU ≥ 0.5) | Polyp frames with any box | "
         "False injections (polyp-free frames with a box) |", "|---|---|---|---|"]
    for k, r in res.items():
        c = r["accuracy"]
        L.append(f"| {names[k]} | {c['seg_correct_box']} / {c['seg_frames']} | {c['polyp_any_box']} / "
                 f"{c['polyp_frames']} | {c['false_injection']} / {c['negative_frames']} |")
    L += ["", "## Uplink bandwidth (board → edge)", "",
          "| Scenario | Bytes per frame (mean) | Frames sent up | " +
          " | ".join(f"kbit/s at {fps} fps (BLE load)" for fps in a.fps) + " |",
          "|---|---|---|" + "---|" * len(a.fps)]
    for k, r in res.items():
        L.append(f"| {names[k]} | {r['uplink_B_per_frame']:,.0f} | {r['frames_sent_up']} | " + " | ".join(
            f"{b['uplink_kbps']:.1f} ({100 * b['ble_load']:.1f} %)" for b in r["battery"].values()) + " |")
    L += ["", "## End-to-end latency with the command back to the board", "",
          "Frame fully received on the board → injection command received on the board (only frames that end "
          "with a box). USB: measured, one board clock. BLE: board compute + modelled uplink + measured edge time "
          "+ modelled command downlink. The drug simulation is not included (separate stage, see RESULTS.md). "
          "C+ is combined from the B and C runs (C's loop + B's classifier time), not one run.", "",
          "| Scenario | Board compute p50 / p95 ms | Edge YOLO26n p50 ms | **E2E USB p50 / p95 ms** (measured) | "
          "**E2E BLE p50 / p95 ms** (modelled) | n |", "|---|---|---|---|---|---|"]
    for k, r in res.items():
        b, ed, u, bl = r["board_ms"], r["edge_ms"], r["usb_e2e_ms"], r["ble_e2e_ms"]
        fm = lambda p: f"{p['p50']:.1f} / {p['p95']:.1f}" if p else "–"  # noqa: E731
        L.append(f"| {names[k]} | {fm(b)} | {ed['p50']:.1f} | {fm(u)} | {fm(bl)} | {u['n'] if u else 0} |"
                 if ed else f"| {names[k]} | {fm(b)} | – | {fm(u)} | {fm(bl)} | {u['n'] if u else 0} |")
    L += ["", "## Energy on the board (estimate) and battery life", "",
          f"Per frame, mean over the test set, in µJ. Battery: {a.battery_mah:.0f} mAh at 3.0 V, chip only "
          "(no regulator losses, no camera link: receiving the frame from the pill costs the same in every "
          f"scenario). Idle between frames {I_MA['idle'] * 1000:.1f} µA.", "",
          "| Scenario | CPU | Axon | Radio TX (uplink) | Radio RX (command) | **Total µJ/frame** | " +
          " | ".join(f"avg mA / battery h at {fps} fps" for fps in a.fps) + " |",
          "|---|---|---|---|---|---|" + "---|" * len(a.fps)]
    for k, r in res.items():
        e = r["energy_mJ_per_frame"]
        L.append(f"| {names[k]} | {1000 * e['cpu']:.0f} | {1000 * e['axon']:.0f} | {1000 * e['tx']:.0f} | "
                 f"{1000 * e['rx_cmd']:.1f} | **{1000 * e['total']:.0f}** | " + " | ".join(
                     f"{b['avg_mA']:.3f} / {b['hours']:,.0f}" for b in r["battery"].values()) + " |")
    path.write_text("\n".join(L) + "\n")


def check():
    """Self-check of the per-frame bookkeeping on two synthetic frames (python compare.py --check)."""
    class A_: ble_kbps, ble_ci_ms, fps, battery_mah = 1000.0, 10.0, [2], 100.0
    row = {"detected": "1", "boxes": "[[0, 0, 10, 10, 0.9]]", "board_ms": "16", "edge_ms": "10", "e2e_ms": "50",
           "t_infer_start_ns": "0", "t_infer_end_ns": "15000000"}
    s = scenario("A", {1: row, 2: {**row, "detected": "0", "boxes": ""}}, {1: 930, 2: 930}, A_)
    x = s["frames"][1]
    assert x["up"] == 1000 and x["ble_e2e"] == 16 + (5 + 8) + 10 + (5 + 0.16)
    assert abs(x["e"]["tx"] - 5.0 * 8 * 3 / 1000) < 1e-12 and abs(x["e"]["axon"] - 2.8 * 15 * 3 / 1000) < 1e-12
    assert s["frames"][2]["up"] == LINE_B and s["frames"][2]["ble_e2e"] is None
    acc = accuracy({1: [[0, 0, 10, 10, 0.9]], 2: [[0, 0, 1, 1, 0.5]]}, {1: 1, 2: 0}, {1: [[0, 0, 10, 11]]})
    assert acc["seg_correct_box"] == 1 and acc["false_injection"] == 1
    print("compare OK")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fwd", type=Path), p.add_argument("--gated", type=Path), p.add_argument("--det", type=Path)
    p.add_argument("--test", type=Path, default=ROOT / "Data/test")
    p.add_argument("--ble-kbps", type=float, default=1300.0, help="BLE 2M PHY application throughput (kbit/s)")
    p.add_argument("--ble-ci-ms", type=float, default=7.5, help="BLE connection interval (ms)")
    p.add_argument("--fps", type=float, nargs="+", default=[2, 6], help="capsule frame rates for bandwidth/battery")
    p.add_argument("--battery-mah", type=float, default=200.0)
    p.add_argument("--out", type=Path, default=ROOT / "results")
    p.add_argument("--check", action="store_true", help="run the self-check and exit")
    a = p.parse_args()
    if a.check:
        return check()
    labels = {int(r["frame_id"]): int(r["polyp"]) for r in csv.DictReader(open(a.test / "labels.csv"))}
    gt = {}
    for r in csv.DictReader(open(a.test / "gt_boxes.csv")):
        gt.setdefault(int(r["frame_id"]), []).append([float(r[k]) for k in ("x_min", "y_min", "x_max", "y_max")])
    jpeg = {i: (a.test / "frames" / f"frame_{i:06d}.jpg").stat().st_size for i in labels}
    A, B, C = rows(a.fwd), rows(a.gated), rows(a.det)
    res = {}
    if A:
        res["A"] = summary(scenario("A", A, jpeg, a), labels, gt, a)
    if B:
        res["B"] = summary(scenario("B", B, jpeg, a), labels, gt, a)
    if C:
        res["C"] = summary(scenario("C", C, jpeg, a), labels, gt, a)
        if B:
            det_ms = {i: (int(r["t_infer_end_ns"]) - int(r["t_infer_start_ns"])) / 1e6 for i, r in C.items()}
            res["C+"] = summary(scenario("C+", C, jpeg, a, det_ms, gate=B), labels, gt, a)
    a.out.mkdir(exist_ok=True)
    (a.out / "comparison.json").write_text(json.dumps(res, indent=2))
    write_md(res, a, a.out / "COMPARISON.md")
    print((a.out / "COMPARISON.md").read_text())


if __name__ == "__main__":
    main()

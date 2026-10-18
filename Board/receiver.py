"""PC side of the board link: replay frames to the nRF54LM20 DK over USB CDC-ACM, read the results, write the CSV.

python Board/receiver.py --port /dev/tty.usbmodemXXXX --frames Anomally_detection/Data/test/frames \
    --out Anomally_detection/runs/pipeline_board/board.csv
python Anomally_detection/pipeline.py --clf Anomally_detection/runs/pipeline_board/board.csv --out runs/pipeline_board

Per frame: send "FRM0" + frame_id + length + JPEG, wait for the reply line, read the echoed JPEG when detected=1 and
check it is byte-identical. Writes the classifier CSV the pipeline reads, in the host clock:
  frame_id,t_rx_board_ns,t_infer_start_ns,t_infer_end_ns,t_sent_ns,score,detected,raw,usb_down_ns,usb_ns
Board timestamps (board clock, us) are only used as differences, anchored at the host send time:
  usb_down_ns = board t_rx - t_first   (frame payload arriving at the board, measured on the board)
  usb_ns      = host round trip - board (t_sent - t_first)
                (result line + echoed frame back to the PC, plus USB scheduling latency of both directions;
                 an upper bound for board -> PC)
  t_rx_board_ns = host send start + usb_down; the other times add the board's own differences to it.
"""
import argparse
import csv
import re
import struct
import time
from pathlib import Path

import serial

COLUMNS = ["frame_id", "t_rx_board_ns", "t_infer_start_ns", "t_infer_end_ns", "t_sent_ns", "score", "detected",
           "raw", "usb_down_ns", "usb_ns"]


def frames(folder):
    """[(frame_id, path)] by id, id = last number in the filename (same rule as Classifier/run_csv.py)."""
    out = sorted((int(re.findall(r"\d+", p.stem)[-1]), p) for p in Path(folder).glob("*.jp*g"))
    if not out:
        raise SystemExit(f"No frames in {folder}")
    return out


def read_reply(port):
    line = port.readline().decode(errors="replace").strip()
    while line and not line.startswith("R,"):  # skip anything that is not a result line
        line = port.readline().decode(errors="replace").strip()
    if not line:
        raise TimeoutError("no reply from the board")
    f = line.split(",")
    return {"fid": int(f[1]), "t_first": int(f[2]), "t_rx": int(f[3]), "t0": int(f[4]), "t1": int(f[5]),
            "t_sent": int(f[6]), "raw": int(f[7]), "score": float(f[8]), "det": int(f[9]), "n": int(f[10])}


def main():
    a = parse_args()
    port = serial.serial_for_url(a.port, timeout=a.timeout)  # baud rate is ignored by CDC-ACM
    port.reset_input_buffer()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    n = hits = bad = 0
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for fid, path in frames(a.frames):
            data = path.read_bytes()
            t_start = time.time_ns()
            port.write(b"FRM0" + struct.pack("<II", fid, len(data)) + data)
            r = read_reply(port)
            echo = port.read(r["n"]) if r["n"] else b""
            t_end = time.time_ns()
            if r["fid"] != fid:
                raise SystemExit(f"reply for frame {r['fid']}, expected {fid}")
            if r["det"] and echo != data:
                bad += 1
            us = 1000
            usb_down = (r["t_rx"] - r["t_first"]) * us
            t_rx = t_start + usb_down
            usb_up = (t_end - t_start) - (r["t_sent"] - r["t_first"]) * us
            w.writerow([fid, t_rx, t_rx + (r["t0"] - r["t_rx"]) * us, t_rx + (r["t1"] - r["t_rx"]) * us,
                        t_rx + (r["t_sent"] - r["t_rx"]) * us, f"{r['score']:.4f}", r["det"], r["raw"],
                        usb_down, usb_up])
            n, hits = n + 1, hits + r["det"]
    print(f"[board] {n} frames, {hits} detected, {bad} echoed frames corrupted -> {a.out}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True, help="CDC-ACM device, e.g. /dev/tty.usbmodem* (macOS), COM5 (Windows); any pyserial URL")
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=2.0, help="seconds to wait for a reply")
    return p.parse_args()


if __name__ == "__main__":
    main()

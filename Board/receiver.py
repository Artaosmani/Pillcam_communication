"""PC side of the board link: replay frames to the nRF54LM20 DK over USB CDC-ACM, read the results, write the CSV.

python Board/receiver.py --port /dev/tty.usbmodemXXXX --frames Anomally_detection/Data/test/frames \
    --out Anomally_detection/runs/pipeline_board/board.csv
python Anomally_detection/pipeline.py --clf Anomally_detection/runs/pipeline_board/board.csv --out runs/pipeline_board
python Board/receiver.py ... --raw     # send 84x84 RGB instead of the JPEG: the board skips the decode

With --raw the PC decodes each 336x336 frame and area-resizes it to 84x84 (exact 4x4 block mean, the resize used in
training) and sends "RAW0" + frame_id + length + 21168 RGB bytes; the board echoes those bytes on a detection.

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
import subprocess
import sys
import time
from pathlib import Path

import serial

RAW_SIDE = 84  # board DEC_SIDE at POLYP_INPUT_SIZE 84

COLUMNS = ["frame_id", "t_rx_board_ns", "t_infer_start_ns", "t_infer_end_ns", "t_sent_ns", "score", "detected",
           "raw", "usb_down_ns", "usb_ns"]


def frames(folder):
    """[(frame_id, path)] by id, id = last number in the filename (same rule as Classifier/run_csv.py)."""
    out = sorted((int(re.findall(r"\d+", p.stem)[-1]), p) for p in Path(folder).glob("*.jp*g"))
    if not out:
        raise SystemExit(f"No frames in {folder}")
    return out


def payload(path, raw):
    """(magic, bytes) to send for one frame."""
    data = path.read_bytes()
    if not raw:
        return b"FRM0", data
    import io
    import numpy as np
    from PIL import Image
    img = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), np.float32)
    k = img.shape[0] // RAW_SIDE
    if img.shape[:2] != (RAW_SIDE * k, RAW_SIDE * k):
        raise SystemExit(f"{path}: {img.shape[:2]} is not a multiple of {RAW_SIDE}")
    return b"RAW0", np.round(img.reshape(RAW_SIDE, k, RAW_SIDE, k, 3).mean((1, 3))).astype(np.uint8).tobytes()


def link_speed():
    """The board's USB link speed as macOS reports it (e.g. "480 Mb/s"), or None (other OS, or board not found).
    The firmware is built for high speed, but the link can still come up at full speed (12 Mb/s, 64-byte packets)."""
    if sys.platform != "darwin":
        return None
    out = subprocess.run(["system_profiler", "SPUSBHostDataType"], capture_output=True, text=True).stdout
    m = re.search(r"Manufacturer: Zephyr Project(?:\n(?!\s*\n).*)*?\n\s*Link Speed: (.+)", out)
    return m.group(1).strip() if m else None


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
    if a.port.startswith("/dev/"):
        speed = link_speed()
        print(f"[board] USB link speed: {speed or 'unknown'}")
        if speed and not speed.startswith("480"):
            print("[board] WARNING: not high speed; replug (directly into the PC, not through a hub) for HS numbers")
    port = serial.serial_for_url(a.port, timeout=a.timeout)  # baud rate is ignored by CDC-ACM
    port.reset_input_buffer()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    n = hits = bad = 0
    prep = []
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for fid, path in frames(a.frames):
            t_prep = time.perf_counter_ns()
            magic, data = payload(path, a.raw)
            prep.append((time.perf_counter_ns() - t_prep) / 1e6)
            t_start = time.time_ns()
            port.write(magic + struct.pack("<II", fid, len(data)) + data)
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
    if a.raw:
        print(f"[board] PC decode + 84x84 resize (not in the CSV): median {sorted(prep)[len(prep) // 2]:.2f} ms")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True, help="CDC-ACM device, e.g. /dev/tty.usbmodem* (macOS), COM5 (Windows); any pyserial URL")
    p.add_argument("--frames", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=2.0, help="seconds to wait for a reply")
    p.add_argument("--raw", action="store_true", help="send 84x84 RGB (RAW0) instead of the JPEG (FRM0)")
    return p.parse_args()


if __name__ == "__main__":
    main()

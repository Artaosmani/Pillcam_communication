"""Self-check for receiver.py against a fake board behind a TCP socket (pyserial socket:// URL, no hardware needed).

python Board/test_receiver.py
"""
import csv
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


def fake_board(server, n_frames, got):
    """Speaks the firmware protocol: frames with an even id are 'polyps' and get echoed (FWD0: every frame), DET0
    gets one box for them, CMD0 an ack 2 ms later on the board clock. Payloads go into `got`."""
    conn, _ = server.accept()
    f = conn.makefile("rwb", buffering=0)
    clock = 0
    while True:
        hdr = f.read(12)
        if not hdr:
            return
        assert hdr[:4] in (b"FRM0", b"RAW0", b"FWD0", b"DET0", b"CMD0")
        fid, n = struct.unpack("<II", hdr[4:])
        data = b""
        while len(data) < n:
            data += f.read(n - len(data))
        got.append((hdr[:4], data))
        if hdr[:4] == b"CMD0":
            f.write(f"A,{fid},{clock - 100000 + 17400},{clock - 100000 + 17500}\n".encode())
            continue
        det = int(fid % 2 == 0 or hdr[:4] == b"FWD0")
        if hdr[:4] == b"DET0":
            t = [clock + k for k in (0, 300, 1300, 15300, 15400, 15500)]
            clock += 100000
            box = ",10,20,30,40,0.900" if det else ""
            f.write(f"D,{fid},{','.join(map(str, t))},{int(bool(box))}{box}\n".encode())
            continue
        t = [clock + k for k in (0, 300, 1300, 15300, 15400)]  # first, rx, infer start/end, sent (us)
        clock += 100000
        time.sleep(0.016)  # really spend the 15.4 ms the timestamps claim
        f.write(f"R,{fid},{t[0]},{t[1]},{t[2]},{t[3]},{t[4]},{50 if det else 0},0.9,{det},{n if det else 0}\n".encode())
        if det:
            f.write(data)


def run(frames, out, *extra):
    server, got = socket.create_server(("127.0.0.1", 0)), []
    threading.Thread(target=fake_board, args=(server, 4, got), daemon=True).start()
    subprocess.run([sys.executable, Path(__file__).with_name("receiver.py"), "--port",
                    f"socket://127.0.0.1:{server.getsockname()[1]}", "--frames", frames, "--out", out, *extra],
                   check=True)
    return list(csv.DictReader(open(out))), got


def main():
    import numpy as np
    from PIL import Image
    with tempfile.TemporaryDirectory() as d:
        frames = Path(d) / "frames"
        frames.mkdir()
        for i in range(1, 5):
            Image.fromarray(np.random.randint(0, 256, (336, 336, 3), np.uint8)).save(frames / f"frame_{i:06d}.jpg")
        det_rows, got_det = run(frames, Path(d) / "board_det.csv", "--det")
        assert [m for m, _ in got_det] == [b"DET0", b"DET0", b"CMD0", b"DET0", b"DET0", b"CMD0"], got_det
        assert len(got_det[0][1]) == 160 * 160 * 3
        assert [int(r["n_boxes"]) for r in det_rows] == [0, 1, 0, 1]
        assert det_rows[1]["boxes"] == "[[21.0, 42.0, 63.0, 84.0, 0.9]]"  # 160 px -> 336 px
        assert float(det_rows[1]["e2e_ms"]) == 17.2 and float(det_rows[1]["board_ms"]) == 15.1
        fwd_rows, got_fwd = run(frames, Path(d) / "board_fwd.csv", "--fwd")
        assert all(m == b"FWD0" for m, _ in got_fwd) and all(int(r["detected"]) for r in fwd_rows)
        rows, got = run(frames, Path(d) / "board.csv")
        assert all(m == b"FRM0" for m, _ in got)
        _, got_raw = run(frames, Path(d) / "board_raw.csv", "--raw")
        img = np.asarray(Image.open(frames / "frame_000001.jpg").convert("RGB"), np.float32)
        want = np.round(img.reshape(84, 4, 84, 4, 3).mean((1, 3))).astype(np.uint8).tobytes()
        assert got_raw[0] == (b"RAW0", want) and all(len(b) == 84 * 84 * 3 for _, b in got_raw)
    assert [int(r["detected"]) for r in rows] == [0, 1, 0, 1], rows
    r = rows[0]
    assert int(r["usb_down_ns"]) == 300_000
    assert int(r["t_infer_end_ns"]) - int(r["t_infer_start_ns"]) == 14_000_000  # 14 ms on the board clock
    assert int(r["t_sent_ns"]) - int(r["t_rx_board_ns"]) == 15_100_000
    assert all(int(r["usb_ns"]) >= 0 for r in rows)
    print("receiver OK")


if __name__ == "__main__":
    main()

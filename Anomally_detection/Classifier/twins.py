"""Find Kvasir v1 polyp images that are (cropped) copies of Kvasir-SEG images, so they can inherit its split.

python Classifier/twins.py [--dir Data/kvasir/kvasir-dataset/polyps] [--out labels/seg_twins.json]

Kvasir-SEG images are Kvasir polyp frames with the black border cropped off, so an image hash misses them. Here the
whole Kvasir-SEG image is slid over each v1 image (1/4 scale, grayscale) and the RMS pixel difference is taken at the
best offset, ignoring the template's black pixels (border, blacked-out scope inset). A copy gives JPEG noise only; a
different image gives a large difference. Writes {v1 filename: [best Kvasir-SEG stem, rms]} for every image;
data.py treats rms <= twin_max_rms as the same image. Takes a few minutes; rerun only if the folders change.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from common import resolve

SCALE = 4
_templates = None


def small(path, width=None):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if width:  # v1 has 720×576 and a few larger frames; Kvasir-SEG was cut from the 720-wide version
        g = cv2.resize(g, (width, round(g.shape[0] * width / g.shape[1])), interpolation=cv2.INTER_AREA)
    return cv2.resize(g, (g.shape[1] // SCALE, g.shape[0] // SCALE), interpolation=cv2.INTER_AREA)


def init(seg_dir):
    global _templates
    _templates = []
    for p in sorted(Path(seg_dir).glob("*.jpg")):
        g = small(p)
        _templates.append((p.stem, g, (g > 20).astype("uint8")))


def best_match(path):
    best = (None, float("inf"))
    for g in (small(path), small(path, 720)):  # native size and 720 wide: Kvasir-SEG was cut from either
        for stem, t, m in _templates:
            if t.shape[0] <= g.shape[0] and t.shape[1] <= g.shape[1]:
                ssd = cv2.matchTemplate(g, t, cv2.TM_SQDIFF, mask=m).min()
                rms = float(np.sqrt(max(ssd, 0) / m.sum()))
                if rms < best[1]:
                    best = (stem, rms)
    return path.name, [best[0], round(best[1], 2)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="Data/kvasir/kvasir-dataset/polyps")
    p.add_argument("--seg", default="Data/Kvasir-SEG/images")
    p.add_argument("--out", default="labels/seg_twins.json")
    a = p.parse_args()
    files = sorted(resolve(a.dir).glob("*.jpg"))
    with ProcessPoolExecutor(initializer=init, initargs=(str(resolve(a.seg)),)) as ex:
        out = dict(ex.map(best_match, files, chunksize=8))
    resolve(a.out).write_text(json.dumps(out, indent=0, sort_keys=True))
    print(f"[twins] {len(out)} images -> {a.out}")


if __name__ == "__main__":
    main()

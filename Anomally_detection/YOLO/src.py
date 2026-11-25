"""Shared paths + dataset build. Builds Data/yolo-split from the Kvasir-SEG split plus the no-polyp images in the
shared split file (Data/splits.json, written by Classifier/data.py), so YOLO26n and the classifier see the same
train/val/test images.

python YOLO/src.py [--config Configs/default.yml]   # (re)build Data/yolo-split and write data.yaml

Positives: Data/kvasir-seg-split/{images,labels}/{split}. Negatives: background images with an empty label file, so
false positives are trained against and measured. train/val get `negatives.ratio` × positives (seeded sample), test
gets every test negative. data.yaml also has `test_pos` (Kvasir-SEG test only) for the box metrics on polyp frames.
Files are symlinks; nothing is copied.
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "Configs/default.yml"
SPLITS = ("train", "val", "test")


def resolve(path):
    """Config paths are relative to the project root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def data_yaml_path(cfg):
    """The dataset description file ultralytics reads: where the images are, which splits exist, class names."""
    return resolve(cfg["paths"]["dataset"]) / "data.yaml"


def prepare(cfg):
    """Rebuild Data/yolo-split (symlinks) and write data.yaml with absolute paths."""
    src, out = resolve(cfg["paths"]["source"]), resolve(cfg["paths"]["dataset"])
    neg_cfg = cfg["negatives"]
    shared = json.loads(resolve(neg_cfg["splits"]).read_text())
    rng = random.Random(cfg["train"]["seed"])
    if out.exists():
        shutil.rmtree(out)
    for split in SPLITS:
        images = sorted((src / "images" / split).glob("*.jpg"))
        labels = {p.stem for p in (src / "labels" / split).glob("*.txt")}
        if not images or {p.stem for p in images} != labels:
            raise SystemExit(f"{src}/{{images,labels}}/{split}: missing images or unpaired labels")
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        for p in images:
            (out / "images" / split / p.name).symlink_to(p)
            (out / "labels" / split / f"{p.stem}.txt").symlink_to(src / "labels" / split / f"{p.stem}.txt")

        neg = [ROOT / p for p, y in shared[split] if y == 0]
        if split != "test":
            neg = rng.sample(neg, min(len(neg), round(neg_cfg["ratio"] * len(images))))
        for p in neg:
            name = f"neg_{p.parent.name.replace(' ', '_')}_{p.stem}"
            (out / "images" / split / f"{name}{p.suffix}").symlink_to(p)
            (out / "labels" / split / f"{name}.txt").touch()
        print(f"[data] {split:5s}: {len(images)} polyp + {len(neg)} background images")
    names = {s: str(out / "images" / s) for s in SPLITS}
    data_yaml_path(cfg).write_text(yaml.safe_dump(
        {**names, "test_pos": str(src / "images" / "test"), "names": {0: "polyp"}}, sort_keys=False))
    print("[data] ->", data_yaml_path(cfg))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare(yaml.safe_load(p.parse_args().config.read_text()))

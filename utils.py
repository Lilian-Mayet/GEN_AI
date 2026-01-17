from pathlib import Path
import random

def make_splits(images_dir: Path, out_dir: Path, val_ratio=0.02, seed=42):
    imgs = sorted([p.name for p in images_dir.glob("*.png")])
    random.Random(seed).shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_ratio))
    val = imgs[:n_val]
    train = imgs[n_val:]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.txt").write_text("\n".join(train), encoding="utf-8")
    (out_dir / "val.txt").write_text("\n".join(val), encoding="utf-8")

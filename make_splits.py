import argparse
from pathlib import Path
from utils import make_splits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, required=True)
    ap.add_argument("--val_ratio", type=float, default=0.02)
    args = ap.parse_args()

    processed = Path(args.processed_dir)
    make_splits(processed / "images", processed.parent / "splits", val_ratio=args.val_ratio)

if __name__ == "__main__":
    main()

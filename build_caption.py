import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

BASE_PREFIX = "pixel sprite, monster, front view"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", type=str, default="data/processed",
                    help="Directory with processed data (default: data/processed)")
    ap.add_argument("--prefix", type=str, default=BASE_PREFIX,
                    help=f"Prefix for captions (default: '{BASE_PREFIX}')")
    args = ap.parse_args()

    processed = Path(args.processed_dir)
    meta = pd.read_csv(processed / "metadata.csv")

    cap_dir = processed / "captions"
    cap_dir.mkdir(parents=True, exist_ok=True)

    def tok(t: str) -> str:
        t = str(t).strip().lower()
        if not t or t == "nan":
            return ""
        return f"type_{t}"

    for _, r in tqdm(meta.iterrows(), total=len(meta)):
        t1 = tok(r.get("type1", ""))
        t2 = tok(r.get("type2", ""))

        tokens = [args.prefix]
        if t1:
            tokens.append(t1)
        if t2:
            tokens.append(t2)

        caption = ", ".join(tokens)
        out_txt = cap_dir / Path(r["image"]).with_suffix(".txt").name
        out_txt.write_text(caption, encoding="utf-8")

    print(f"Captions written to: {cap_dir}")

if __name__ == "__main__":
    main()

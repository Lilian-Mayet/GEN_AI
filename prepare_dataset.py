import os
import argparse
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm

def center_pad_to_square(img: Image.Image, fill=(0,0,0,0)) -> Image.Image:
    # Works for RGBA or RGB. If RGB, fill will be truncated.
    w, h = img.size
    m = max(w, h)
    new = Image.new(img.mode, (m, m), fill if img.mode == "RGBA" else fill[:3])
    new.paste(img, ((m - w) // 2, (m - h) // 2))
    return new

def process_one(in_path: Path, out_path: Path, size: int):
    img = Image.open(in_path).convert("RGBA")
    # optional: if there is transparency, keep it.
    img = center_pad_to_square(img, fill=(0,0,0,0))
    img = img.resize((size, size), Image.NEAREST)  # NEAREST preserves pixel art edges
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="../downloads", help="Directory with raw pokemon sprites (default: ../downloads)")
    ap.add_argument("--out_dir", type=str, default="data/processed", help="Output directory for processed data (default: data/processed)")
    ap.add_argument("--size", type=int, default=128, help="Target size for resized images (default: 128)")
    ap.add_argument("--types_csv", type=str, default="../downloads/pokemon_data.csv",
                    help="CSV mapping pokemon_name -> type1,type2 (default: ../downloads/pokemon_data.csv)")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    img_out = out_dir / "images"
    cap_out = out_dir / "captions"
    img_out.mkdir(parents=True, exist_ok=True)
    cap_out.mkdir(parents=True, exist_ok=True)

    types_df = pd.read_csv(args.types_csv)

    # Expect columns: pokemon_name,type1,type2 (type2 optional)
    # Example row: bulbasaur,grass,poison
    types_map = {}
    for _, r in types_df.iterrows():
        name = str(r["pokemon_name"]).strip().lower()
        t1 = str(r["type1"]).strip().lower()
        t2 = str(r.get("type2", "")).strip().lower() if "type2" in types_df.columns else ""
        types_map[name] = (t1, t2 if t2 and t2 != "nan" else "")

    rows = []
    for pokemon_dir in sorted([p for p in raw_dir.iterdir() if p.is_dir()]):
        pokemon = pokemon_dir.name.strip().lower()
        if pokemon not in types_map:
            # If some are missing, still process but leave types empty
            t1, t2 = "", ""
        else:
            t1, t2 = types_map[pokemon]

        for i, file in enumerate(sorted(pokemon_dir.glob("*"))):
            if file.suffix.lower() not in [".png", ".jpg", ".jpeg", ".webp"]:
                continue
            out_name = f"{pokemon}__{i:04d}.png"
            out_path = img_out / out_name
            process_one(file, out_path, args.size)

            rows.append({
                "image": out_name,
                "pokemon_name": pokemon,
                "type1": t1,
                "type2": t2
            })

    meta = pd.DataFrame(rows)
    meta_path = out_dir / "metadata.csv"
    meta.to_csv(meta_path, index=False)
    print(f"Saved: {meta_path} with {len(meta)} rows")

if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--ckpt_dir", type=str, required=True, help="Folder with config.json + diffusion_pytorch_model.safetensors")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--out", type=str, default="outputs/samples/out.png")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--size", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    ).to(device)

    # Load trained UNet checkpoint (full model)
    unet = UNet2DConditionModel.from_pretrained(
        args.ckpt_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        device_map=None,
    ).to(device)
    pipe.unet = unet

    g = torch.Generator(device=device).manual_seed(args.seed)
    img = pipe(
        prompt=args.prompt,
        height=args.size,
        width=args.size,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        generator=g,
    ).images[0]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

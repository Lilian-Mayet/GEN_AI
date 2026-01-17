import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig
from transformers import CLIPTokenizer


class SpriteDataset(Dataset):
    def __init__(self, images_dir: Path, captions_dir: Path, file_list: list[str], size: int = 128):
        self.images_dir = images_dir
        self.captions_dir = captions_dir
        self.files = file_list
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img_path = self.images_dir / name
        cap_path = self.captions_dir / Path(name).with_suffix(".txt").name

        # Load image (RGB) -> tensor in [-1, 1]
        img = Image.open(img_path).convert("RGB")

        # PIL -> torch tensor (C,H,W), float32
        x = torch.from_numpy(
            (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
             .view(img.size[1], img.size[0], 3)
             .numpy())
        ).float() / 255.0
        x = x.permute(2, 0, 1).contiguous()
        x = x * 2.0 - 1.0

        caption = cap_path.read_text(encoding="utf-8").strip()
        return x, caption


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    captions = [b[1] for b in batch]
    return images, captions


def set_lora(pipe, rank: int = 16):
    """
    Diffusers 0.36 + PEFT: on ajoute un adapter LoRA dans l'UNet.
    """
    if not hasattr(pipe.unet, "add_adapter"):
        raise RuntimeError("pipe.unet.add_adapter() not found. Check diffusers/peft install.")

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        lora_dropout=0.0,
        bias="none",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    pipe.unet.add_adapter(lora_config)
    return pipe


def get_lora_params(unet) -> list[torch.nn.Parameter]:
    trainable = []
    for name, p in unet.named_parameters():
        if "lora" in name.lower():
            p.requires_grad = True
            if p.dtype != torch.float32:
                p.data = p.data.float()
            trainable.append(p)
        else:
            p.requires_grad = False
    if len(trainable) == 0:
        raise RuntimeError("No LoRA params found (expected 'lora' in parameter names).")
    return trainable



def save_lora_weights(pipe, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Sauvegarde UNIQUEMENT la LoRA (léger)
    pipe.save_lora_weights(out_dir) 



def load_file_list(list_path: Path):
    return [l.strip() for l in list_path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--processed_dir", type=str, required=True)
    ap.add_argument("--splits_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--save_every", type=int, default=0,
                help="Save LoRA every N epochs (0 = only final).")
    ap.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processed = Path(args.processed_dir)
    splits = Path(args.splits_dir)
    out_dir = Path(args.output_dir)

    images_dir = processed / "images"
    captions_dir = processed / "captions"

    train_files = load_file_list(splits / "train.txt")

    dtype = torch.float16 if args.mixed_precision in ["fp16", "bf16"] else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=dtype,  # (warning de torch_dtype deprecated est OK, mais ça marche)
        safety_checker=None,
        requires_safety_checker=False,
    )

    # Add LoRA
    pipe = set_lora(pipe, rank=args.rank)
    pipe.to(device)

    # Force LoRA weights to fp32 to work with GradScaler (avoid FP16 grads unscale error)
    for name, p in pipe.unet.named_parameters():
        if "lora" in name.lower():
            p.data = p.data.float()


    # Freeze non-LoRA parts
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # Only LoRA params trainable
    lora_params = get_lora_params(pipe.unet)

    tokenizer: CLIPTokenizer = pipe.tokenizer
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    ds = SpriteDataset(images_dir, captions_dir, train_files, size=args.size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

    opt = torch.optim.AdamW(lora_params, lr=args.lr)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.mixed_precision == "fp16"))

    pipe.unet.train()
    global_step = 0

    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"epoch {epoch+1}/{args.epochs}")
        opt.zero_grad(set_to_none=True)

        for step, (images, captions) in enumerate(pbar):
            images = images.to(device, dtype=dtype)  # VAE encode préfère fp32 input

            # Tokenize
            text_inputs = tokenizer(
                captions,
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.to(device)

            with torch.no_grad():
                encoder_hidden_states = pipe.text_encoder(text_inputs)[0]
                images = images.to(dtype=pipe.vae.dtype)  
                latents = pipe.vae.encode(images).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor
                latents = latents.to(device, dtype=dtype)


            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],),
                device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.cuda.amp.autocast(enabled=(args.mixed_precision == "fp16")):
                model_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = torch.nn.functional.mse_loss(model_pred, noise)

            loss = loss / args.grad_accum
            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})

        if args.save_every and ((epoch + 1) % args.save_every == 0):
            save_lora_weights(pipe, out_dir / f"checkpoint_epoch_{epoch+1}")

    # Final save
    save_lora_weights(pipe, out_dir / "final")
    print(f"Done. LoRA saved to: {out_dir}")


if __name__ == "__main__":
    main()

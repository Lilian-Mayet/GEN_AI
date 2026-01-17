import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from pathlib import Path

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CKPT_DIR = Path("lora_checkpoint")  # <- mets ici ton dossier checkpoint
OUT = "result.png"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype,
    safety_checker=None,
    requires_safety_checker=False,
).to(device)

# Charger l'UNet entraîné (checkpoint complet)
unet = UNet2DConditionModel.from_pretrained(
    str(CKPT_DIR),
    torch_dtype=dtype,
    low_cpu_mem_usage=False,   # IMPORTANT: évite les meta tensors
    device_map=None            # IMPORTANT: évite l'initialisation meta via accelerate
).to(device)
pipe.unet = unet

prompt = "pixel sprite, monster, front view, type_ice"
img = pipe(prompt, height=128, width=128, num_inference_steps=30, guidance_scale=7.0).images[0]
img.save(OUT)
print("Saved:", OUT)

import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline
from peft import PeftModel

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
LORA_DIR = Path("lora_final")   # dossier qui contient adapter_config.json + adapter_model.safetensors
OUT = "result.png"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=dtype,
    safety_checker=None,
    requires_safety_checker=False,
).to(device)

# Charge l'UNet avec l'adapter LoRA (PEFT) par-dessus
pipe.unet = PeftModel.from_pretrained(pipe.unet, str(LORA_DIR)).to(device)

prompt = "pixel sprite, monster, front view, type_dragon, type_ice"

image = pipe(
    prompt=prompt,
    height=128,
    width=128,
    num_inference_steps=30,
    guidance_scale=7.0,
).images[0]

image.save(OUT)
print("Saved:", OUT)

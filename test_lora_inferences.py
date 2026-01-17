import torch
from diffusers import StableDiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device=="cuda" else torch.float32

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype,
    safety_checker=None,
    requires_safety_checker=False
).to(device)

prompt = "pixel sprite, monster, front view, type_dragon, type_ice"
g = torch.Generator(device=device).manual_seed(123)

img_base = pipe(prompt, height=128, width=128, num_inference_steps=30, guidance_scale=7.0, generator=g).images[0]
img_base.save("base.png")

pipe.load_lora_weights("lora_checkpoint")   # le dossier qui contient pytorch_lora_weights.safetensors
pipe.fuse_lora()

g = torch.Generator(device=device).manual_seed(123)
img_lora = pipe(prompt, height=128, width=128, num_inference_steps=30, guidance_scale=7.0, generator=g).images[0]
img_lora.save("lora.png")
print("saved base.png & lora.png")

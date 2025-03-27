import torch
from diffusers import DiffusionPipeline
import sys 
print(sys.path)

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
lora_model_path = "lora/ColoringBookRedmond-ColoringBook-ColoringBookAF.safetensors"

base = DiffusionPipeline.from_pretrained(model_id,
                                             torch_dtype=torch.float16,
                                             variant='fp16',
                                             use_safetensors=True)

base.load_lora_weights(lora_model_path)
base.enable_model_cpu_offload()


image = base(prompt='simple cow coloring book', num_inference_steps=20).images[0]
print("1")
image.save("output_lora.jpg")
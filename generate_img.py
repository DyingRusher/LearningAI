import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler ,DiffusionPipeline
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image

# Remaining
# use lora
# use refiner 
model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
height = 1024
width = 1024
num_inference_steps = 25
denoising_factor = 0.8 # similarity between base and refined model

# base model step = 0.8*25
# refined model step = 0.2*25

image = pipe(prompt).images[0]

image.save("astronaut_rides_horse_simple.png")

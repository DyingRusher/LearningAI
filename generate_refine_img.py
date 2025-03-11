import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler ,DiffusionPipeline
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image



image = load_image("astronaut_rides_horse_simple.png")
prompt = "a photo of an astronaut riding a horse on mars"


refined_model = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-refiner-1.0',
    widht=512,
    height=512,
    variant='fp16'
)

refined_model = refined_model.to("cuda")

# img_l = pipe(prompt=prompt,num_inference_steps=num_inference_steps,denoising_strength=denoising_factor,output_type='latent').images[0] # out will be vector of vector with [-1,1]

# torch.cuda.empty_cache()

# img_l = img_l.unsqueeze(0)

final_img = refined_model(prompt=prompt,num_inference_steps = 25 ,denoising_strength = 0.8,image = image).images[0]

final_img.save("refined_img")
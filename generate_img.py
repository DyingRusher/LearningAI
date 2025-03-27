import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Model and LoRA paths
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
lora_model_path = "lora/ColoringBookRedmond-ColoringBook-ColoringBookAF.safetensors"

# Load the base model
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)

# Set the scheduler for better results
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Load the LoRA weights - requires PEFT
pipe.load_lora_weights(lora_model_path)

# Enable CPU offloading for memory efficiency
pipe.enable_model_cpu_offload()

# Set your prompt - add any trigger words that might be specific to this LoRA
prompt = "a photo of an astronaut riding a horse on mars, coloring book style"

# Generate the image
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=25,
    guidance_scale=7.5,
    cross_attention_kwargs={"scale": 0.8}  # Adjust LoRA strength between 0-1
).images[0]

# Save the result
image.save("astronaut_coloring_book.png")
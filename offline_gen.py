import torch
from diffusers import DPMSolverMultistepScheduler ,DiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# Remaining
# use lora
# use refiner

def generate_img_off(prompt,model_name =  "stabilityai/stable-diffusion-2-1"):

    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    num_inference_steps = 25

    image = pipe(prompt).images[0]
    return image



def generate_text_off(prompt,model_name="Qwen/Qwen2.5-3B-Instruct"):

    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    
    messages = [
    # {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return  response


def main():
    pass
if __name__ == "__main__":
    main()
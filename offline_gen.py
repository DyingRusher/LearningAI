from qwen_vl_utils import process_vision_info
from diffusers import DiffusionPipeline
from transformers import AutoTokenizer,AutoModelForCausalLM
import torch

torch.classes.__path__ = []
# Remaining
# use refiner

def generate_text_from_img_off(model,processor,image,prompt="Describe this image."):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                }, 
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


def load_mode_img_0ff(model_path="stabilityai/stable-diffusion-xl-base-1.0"):
    print("load _ model")
    base = DiffusionPipeline.from_pretrained(model_path,
                                             torch_dtype=torch.float16,
                                             variant='fp16',
                                             use_safetensors=True)
    base.enable_model_cpu_offload()
    return base

def generate_img_off(prompt,base):
    image = base(prompt=prompt, num_inference_steps=20).images[0]
    return image

def load_generate_img_off(prompt,model_name =  "stabilityai/stable-diffusion-xl-base-1.0",lora_path = None):

    base = DiffusionPipeline.from_pretrained(model_name,
                                             torch_dtype=torch.float16,
                                             variant='fp16',
                                             use_safetensors=True)

    base.enable_model_cpu_offload()
    if lora_path:
        base.load_lora_weights(lora_path)
        image = base(prompt=prompt + " coloring book", num_inference_steps=20).images[0]

    else:
        image = base(prompt=prompt, num_inference_steps=20).images[0]

    return image

def load_generate_text_off(prompt,model_name="Qwen/Qwen2.5-3B-Instruct"):

    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cuda"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name,device_map="cuda")
    # model.to("cuda")
    
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
    print("11")
    generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024
        )
    print("12")
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return  response

def main():
    pass

if __name__ == "__main__":
    main()
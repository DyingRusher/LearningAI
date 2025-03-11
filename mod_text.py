from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import streamlit as st

#remaining
# 1 use thread to achieve paraller processing
# 2 stream output

    
@st.cache_resource() # now it will not load again and again
def load_model(model_name="Qwen/Qwen2-VL-2B-Instruct"):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    return model

def generate_text(model,prompt):

    print("1",prompt)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    print("2",prompt)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]
    print("3",prompt)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    print("4",prompt)
    image_inputs, video_inputs = process_vision_info(messages)
    print("5",prompt)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    print("6",prompt)
    inputs = inputs.to("cuda")
    print("7",prompt)
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    print("8",prompt)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print("9",output_text)
    return output_text[0]

def main():
    
    st.title("generate text")

    prompt = st.text_input("prompt",placeholder="Enter your prompt") # value = default
    model = load_model()

    if st.button("Generate Text"):
        with st.spinner("wait"):
            res = generate_text( model ,prompt)
            st.write(res)

if __name__ == "__main__":
    main()

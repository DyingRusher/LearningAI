import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# print(sys.path)
from offline_gen import *
import streamlit as st

st.title("Sticker Generator")

model_id = "stabilityai/stable-diffusion-xl-base-1.0"

@st.cache_resource()
def load_model():
    base = DiffusionPipeline.from_pretrained(model_id,
                                                torch_dtype=torch.float16,
                                                variant='fp16',
                                                use_safetensors=True)

    base.load_lora_weights('lora/StickersRedmond.safetensors')
    base.enable_model_cpu_offload()
    return base

base = load_model()
user_input = st.text_input("Enter prompt",value="cat")

lora_trigger = "sticker"
prompt = user_input + " ," + lora_trigger

num_img = st.slider("number of images you want to generate",1,10,2)

if st.button("Generate"):
    with st.spinner('generating pages'):
        progress_bar  = st.progress(0)
        print(prompt)
        for i in range(num_img):
            if i % 2 == 0:
                cols = st.columns(2)

            image_g = base(prompt=prompt, num_inference_steps=20).images[0]
            print(image_g)
            with cols[i%2]:
                st.image(image_g,use_column_width=True)
            
            progress = (i+1)/num_img
            progress_bar.progress(int(progress*100))

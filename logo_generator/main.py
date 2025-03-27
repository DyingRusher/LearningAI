import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from offline_gen import *
import streamlit as st

st.title("Logo generator")

model_name = "stabilityai/stable-diffusion-xl-base-1.0"
lora_name = "lora/logomkrdsxl.safetensors"

@st.cache_resource()
def load_model():
    base = DiffusionPipeline.from_pretrained(model_name ,torch_dtype=torch.float16,
                                                variant='fp16',
                                                use_safetensors=True)
    base.load_lora_weights(lora_name)
    base.enable_model_cpu_offload()

    return base

base = load_model()

user_in = st.text_input("Enter prompt for logo",value = "Transform Tuning")
lora_trigger = 'logo of '
prompt = lora_trigger + user_in

num_img = st.slider("number of images you want to generate",1,10,2)

c1,c2,c3,c4,c5 = st.columns(5)
with c1:
    check1 = st.checkbox("colorful")
with c2:
    check2 = st.checkbox("black and white")
with c3:
    check3 = st.checkbox("Minimalistic")
with c4:
    check4 = st.checkbox("Detailed")
with c5:
    check5 = st.checkbox("Circle")

if check1:
    prompt = prompt + ", colorful"
if check2:
    prompt = prompt + ", black and white"
if check3:
    prompt = prompt + ", minimalistic"
if check4:
    prompt = prompt + ", detailed"
if check5:
    prompt = prompt + ", circle"

if st.button("Generate image"):
    with st.spinner('Creating image'):
        pro_bar = st.progress(0)

        for i in range(num_img):
            if i%2 == 0:
                cols = st.columns(2)
            
            image_generated = base(prompt=prompt, num_inference_steps=20).images[0]

            with cols[i%2]:
                st.image(image_generated,use_column_width=True)

            progress = (i+1)/num_img
            pro_bar.progress(int(progress*100))
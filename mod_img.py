
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

import streamlit as st

@st.cache_resource()
def load_model(model_name= "stabilityai/stable-diffusion-2-1"):

    
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    return pipe

def generate_img_from_text(model,prompt):

    image = model(prompt).images[0]
    return image

def main():

    st.title("making modules")
    prompt = st.text_input("",placeholder="Enter your prompt") # value = default
    model = load_model()

    if st.button("Generate Image"):
        with st.spinner("please wait"):
            image = generate_img_from_text(model,prompt)
            st.image(image)

if __name__ == "__main__":
    main()
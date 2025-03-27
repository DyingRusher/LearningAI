import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from offline_gen import *
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
import streamlit as st

st.title("Caption Generator")

@st.cache_resource()
def load_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map='cuda'
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct",min_pixels=256*256,max_pixels=512*512,device_map='cuda')

    return model,processor

model , processor = load_model()

uploaded_image = st.file_uploader("choose a photo from your pc",type=['.png,.jpg,.jpeg'])

if st.button("Generate Semantic"):
    with st.spinner("Generating Semantics.."):
        col1,col2 = st.columns(2)

        with col1:
            st.image(uploaded_image,width=300)

        with col2:
            pil_image = Image.open(uploaded_image)
            semantics = generate_text_from_img_off(model,processor,pil_image)
            st.subheader("Description")
            st.text(semantics)

            prompt = "Based on the image description, generate 3 captions for instagram" \
                     "Add Emojis and hashtags " \
                     "Here is the description " + semantics
    
            caption = load_generate_text_off(prompt=prompt)
            st.subheader("caption")
            st.text(caption)


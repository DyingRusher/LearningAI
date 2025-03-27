import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from offline_gen import *
from PIL import Image
import uuid
import os
from transformers import Qwen2_5_VLForConditionalGeneration,AutoProcessor
import streamlit as st
import numpy as np
import cv2


def rename_save_img(img,save_path,semantics):
    semantics = semantics.strip()
    semantics = semantics.replace(" ","_")
    semantics = semantics.replace(".","")
    new_file_name = f"{uuid.uuid4()}_{semantics}.jpg"
    new_save_path = os.path.join(Path(save_path),new_file_name)

    print("new   path",new_save_path,img)
    cv2.imwrite(new_save_path,img)
    print("saved")
    return new_save_path

def save_and_process_files(uploaded_files,upload_dir):
    # print("uploaded_f",uploaded_files)
    progress_bar = st.progress(0)

    count = st.empty()
    for i,uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file)
        semantics = generate_text_from_img_off(model,processor,image,prompt="Describe image in 10 words")
        image = cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
        # print("img",img)
        rename_save_img(image,upload_dir,semantics)
        progress_bar.progress(int((i+1)/len(uploaded_files)*100))
        count.text(f"{i+1}/{len(uploaded_files)} images processed")


def get_image_files(upload_dir):
    return [os.path.join(upload_dir,filename) for filename in os.listdir(upload_dir) if filename.endswith(('.png','.jpg','.jpeg'))]

def filter_image(search_query,upload_dir):
    image_files = get_image_files(upload_dir)
    if search_query:
        keywords = search_query.lower().split()

        filter_files = [file for file in image_files if all(keyword in Path(file).stem.lower() for keyword in keywords)]
        return filter_files
    else:
        return image_files

def dispaly_images_in_grid(images_files):

    if images_files:
        num_cols = 3
        cols = st.columns(num_cols)
        for i,file_path in enumerate(images_files):
            image = Image.open(file_path)
            cols[i%num_cols].image(image,use_container_width=True)

st.title("Google Photo Replica")

@st.cache_resource()
def load_model():
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map='cuda'
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct",min_pixels=256*256,max_pixels=512*512,device_map='cuda')
    

    return model,processor

model , processor = load_model()

upload_dir = 'uploaded_img'
os.makedirs(upload_dir,exist_ok=True)

if 'file_uploader_key' not in st.session_state:
    st.session_state['file_uploader_key'] = uuid.uuid4().hex

uploaded_img = st.file_uploader("choose a photo",accept_multiple_files=True,type=['.jpg','.png','.jpeg'],key = st.session_state['file_uploader_key'])

if uploaded_img:
    save_and_process_files(uploaded_img,upload_dir)
    st.session_state['file_uploader_key'] = uuid.uuid4().hex

search_query = st.text_input("Search Images")

filter_file = filter_image(search_query,upload_dir)
dispaly_images_in_grid(filter_file)
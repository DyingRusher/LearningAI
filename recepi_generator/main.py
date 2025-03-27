import sys
from pathlib import Path
# remaining 
# use can generate only one time then restart the server
sys.path.append(str(Path(__file__).resolve().parents[1]))

from offline_gen import *
import streamlit as st

st.title("AI Recipe Generator")

output_format = ("""
                    <h1> Fun Title of recipe </h1>
                    <h1> Table of Contents</h1> <li> links of content </li>
                    <h1> Introduction </h1><p> dish introduction</p>
                    <h1> Country of Origin </h1><p> Country of Origin</p>
                    <h1> Ingredients </h1><li>Ingredients list </li>
                    <h1> Cooking Steps</h1><li>Cooking Steps list </li>
                    <h1> FAQ </h1><p>question answers</p>
                 """)

recipe = st.text_input("Enter the recipe name")

if st.button("Generate Recipe"):
    with st.spinner("Generating Recipe"):
        prompt_text = f" Create a detail cooking recipe for {recipe}."\
                 f" Include preparation steps , ingredients and cooking steps and time."\
                 f" Follow the output format: {output_format}"
        
        res_text = load_generate_text_off(prompt_text)
        st.markdown(res_text,unsafe_allow_html=True)

    with st.spinner("Generating Image"):
        image = load_generate_img_off(recipe + " realistic image")
        st.image(image)
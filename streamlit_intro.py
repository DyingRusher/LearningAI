import streamlit as st

st.title("Hello this is title")

st.header("this is header")

st.subheader("this is subheader")

st.text("this is text")

st.markdown("this is markdown")
st.markdown("## this is markdown header")

st.button("this is button")

st.checkbox("this is checkbox")

st.radio("radio",['1','2','3'])

st.selectbox("select",['1','2','3']) 

st.multiselect("multisellect",['1','2','3'])

st.file_uploader("uploadfile",type=['png','jpg'])

st.color_picker("color picker")

st.date_input("date input")

st.time_input("time input")

st.text_input("Text input",placeholder="enter your age") # value = default

st.number_input("number input",min_value=69,max_value=100,value=71)

st.text_area("this is text area")

st.slider("this is slider",max_value=100,min_value=50,value = 69)


import time
bar = st.progress(100)
for i in range(11):
    time.sleep(0.1)
    bar.progress(100-i*10)

with st.spinner("Wait you fool"):
    time.sleep(0.5)


col1 , col2 = st.columns(2)

with col1:
    st.header("col1")

with col2:
    st.header("col2")

image = st.file_uploader("upload an image",type=['png','jpg'])

if image:
    st.image(image,caption="uploaded image",use_column_width=True)
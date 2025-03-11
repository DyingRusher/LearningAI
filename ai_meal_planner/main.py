import sys
from pathlib import Path
# remaining 
# use can generate only one time then restart the server
sys.path.append(str(Path(__file__).resolve().parents[1]))

from offline_gen import *
import streamlit as st
import json

st.title("Ai meal planner")

c1,c2 = st.columns(2)
with c1:
    gender = st.selectbox('Gender',('Male','Female','other'))
    age = st.number_input('Age',min_value=10,max_value=100)
with c2:
    weight = st.number_input('Weight(kg)',min_value=40,max_value=200)
    height = st.number_input("height(cm)",min_value=140,max_value=250)

aim = st.selectbox('Aim',('Lose weight','Gain weight','Maintain weight'))

user_data = f""" - I am a {gender}
                - My weight is {weight} kg
                - I am {age} years old
                - My height is {height} cm
                - My aim is to {aim}
             """

output_format = """ "range":"Range of ideal weight",
                    "target":"Target weight",
                    "difference":"Weight i need to loose or gain",
                    "bmi":"my BMI",
                    "meal_plan":"Meal plan for 7 days",
                    "total_days":"Total days to reach target weight",
                    "weight_per_week":"Weight to loose or gain per week",
                                    """

prompt = user_data + (" plan a meal plan for 7 days with given information,follow the output format as follows."
                      " Give only json format nothing else ") + output_format

if st.button("Generate Meal plan"):
    with st.spinner("Generating meal plan"):
        print(prompt)
        res_text = generate_text_off(prompt)
        # Check if the string starts with ```json and remove it
        print("res",res_text)
        if res_text.startswith("```json"):
            res_text = res_text.replace("```json\n", "", 1)  # Remove the first occurrence
        if res_text.endswith("```"):
            res_text = res_text.rsplit("```", 1)[0]  # Remove the trailing part

        meal_plan_json = json.loads(res_text)
 
        st.title("Meal Plan")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Range")
            st.write(meal_plan_json["range"])
            st.subheader("Target")
            st.write(meal_plan_json["target"])
        with col2:
            st.subheader("BMI")
            st.write(meal_plan_json["bmi"])
            st.subheader("Days")
            st.write(meal_plan_json["total_days"])
 
        with col3:
            st.subheader(f"{aim}")
            st.write(meal_plan_json["difference"])
            st.subheader("Per week")
            st.write(meal_plan_json["weight_per_week"])
 
        st.subheader("Meal plan for 7 days")
        st.write(meal_plan_json["meal_plan"])
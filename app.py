import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import sklearn

import streamlit as st

import pickle


MODEL_FILE = "./artefacts/logistic_regression.bin"

st.markdown(f"""
<style>
.stApp{{
    background-image: url(https://i.imgur.com/XtLvKZO.jpg);
    background-size: cover;
}}
</style>
""",unsafe_allow_html=True)

st.title("Welcome to the text difficulty classifier!")

#st.markdown(
#""" ## Welcome to the text difficulty classifier.
#"""
#)

#st.image("https://mdpiblog.wordpress.sciforum #.net/wp-content/uploads/sites/4/2018/01/books.jpg", width=400)
    
st.markdown(
""" ### This model predicts the difficulty of a given text according to the reference levels A1, A2, B1, B2, C1, C2 from the Common European Framework of Reference for Languages (CEFR).
    """
)

st.markdown(
""" ##### For predictions regarding a text, please insert it below.
    """
)

#@st.cache
#def load_model():
#    with open(MODEL_FILE, "rb") as file_in:
#            clf_LR = pickle.load(file_in)
#    return clf_LR

#clf_LR = load_model()
 
with open(MODEL_FILE, "rb") as file_in:
    clf_LR = pickle.load(file_in)

txt = st.text_area('Enter text: ')
    

user_input = {
        "txt": str(txt)
        }


text = [str(txt)]
pred = clf_LR.predict(text)
proba = clf_LR.predict_proba(text)

if st.button("Predict"):
    st.write(txt)
    st.success(f'''## This text corresponds to the level {pred} with a probability of: {np.round(proba, 2)}''')
    


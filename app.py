import streamlit as st
import os
from model import predict_disease, load_model
from database import add_crop, get_user_crops, add_user, check_user
from eda import generate_eda
import shutil

MODEL = load_model()

st.set_page_config(page_title="Crop Disease Predictor", layout="wide")

if 'user_id' not in st.session_state:
    st.session_state.user_id = None

def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user_id = check_user(username, password)
        if user_id:
            st.session_state.user_id = user_id
            st.success("Logged in successfully!")
        else:
            st.error("Invalid credentials")
    if st.button("Sign Up"):
        user_id = add_user(username, password)
        st.session_state.user_id = user_id
        st.success("Signed up and logged in!")

def main_page():
    st.title("Crop Disease Predictor")
    uploaded_file = st.file_uploader("Upload an image of a crop leaf", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        filepath = os.path.join("static/uploads", uploaded_file.name)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        prediction = predict_disease(filepath, MODEL)
        st.image(filepath, caption="Uploaded Image", width=300)
        st.write(f"Prediction: {prediction}")
        
        add_crop(st.session_state.user_id, filepath, prediction)
    
    st.subheader("Your Crop History")
    crops = get_user_crops(st.session_state.user_id)
    for crop in crops:
        st.image(crop.image_path, caption=crop.prediction, width=150)
    
    if crops:
        eda_path = generate_eda(crops, st.session_state.user_id)
        if eda_path:
            st.image(eda_path, caption="EDA: Prediction Distribution")

if st.session_state.user_id is None:
    login_page()
else:
    main_page()
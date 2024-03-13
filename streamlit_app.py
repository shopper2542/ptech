import streamlit as st
from roboflow import Roboflow
import tempfile
import numpy as np
import cv2
import os

st.title("Coin Counter 🪙")

st.header("ถ่ายและอัพโหลดภาพเหรียญ")

picture = st.camera_input("")

if picture:
    # Save the captured image to a temporary file
    temp_file_path = tempfile.mkstemp(suffix=".jpg")[1]
    with open(temp_file_path, "wb") as f:
        f.write(picture.getvalue())

    # Initialize Roboflow instance
    rf = Roboflow(api_key="vn8JrzqqTsup7k4rVPhq")
    
    # Specify the workspace and project
    workspace = rf.workspace()
    project = workspace.project("coin-counter-gzvkf")

    # Specify the model
    model = project.version(1).model

    # Predict on the saved image
    prediction = model.predict(temp_file_path).json()
    print(prediction)

    # You can visualize the prediction if needed
    # st.image(prediction)

    # Delete the temporary file
    os.remove(temp_file_path)

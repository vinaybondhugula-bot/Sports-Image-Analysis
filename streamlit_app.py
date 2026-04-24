import streamlit as st
import cv2
import joblib
import json
import numpy as np
from PIL import Image

# 1. Load artifacts (model and dictionary)
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('saved_model.pkl')
        with open("class_dictionary.json", "r") as f:
            class_dict = json.load(f)
        # Reverse the dictionary to get name from index
        class_names = {v: k for k, v in class_dict.items()}
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

model, class_names = load_artifacts()

# 2. Sidebar/Header
st.set_page_config(page_title="Celebrity Classifier", page_icon="⚽")
st.title("Sports Celebrity Image Classifier")
st.markdown("Upload an image of a sports star to see who it is!")

# 3. File Uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display original image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Process for Prediction
    if st.button('Identify Celebrity'):
        if model is not None:
            # Convert PIL image to OpenCV format
            #img_array = np.array(image.convert('RGB'))
            img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            #img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # --- CRITICAL: MATCH THE SIZE USED IN YOUR NOTEBOOK ---
            # If your model was trained on 64x64, change (32, 32) to (64, 64)
            #img_resized = cv2.resize(img_cv, (64, 64))
            img_resized = cv2.resize(img_gray, (64, 64))
            
            # Reshape based on model input (1, flattened_pixels)
            # For 32x32 RGB images, this is 1 x 3072
            img_flattened = img_resized.reshape(1, 64*64*3).astype(float)
            
            # Get Prediction
            prediction = model.predict(img_flattened)[0]
            probs = model.predict_proba(img_flattened)[0]
            
            # 4. Show Results
            name = class_names[prediction].replace('_', ' ').title()
            st.success(f"I am {int(max(probs)*100)}% sure this is **{name}**!")
            
            # Show Probability Chart
            st.write("Confidence Breakdown:")
            chart_data = {class_names[i].title(): float(probs[i]) for i in range(len(probs))}
            st.bar_chart(chart_data)
        else:
            st.error("Model not loaded correctly. Check your .pkl file.")

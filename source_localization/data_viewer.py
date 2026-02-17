import streamlit as st
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Data Viewer", layout="centered")

st.title("Data Viewer")

# Input box for image path
image_path = st.text_input("Enter the path to an image:", placeholder="/path/to/image.jpg")

# Display image if path is provided
if image_path:
    try:
        path = Path(image_path)
        if path.exists():
            image = Image.open(path)
            st.image(image, use_container_width=True)
        else:
            st.error(f"File not found: {image_path}")
    except Exception as e:
        st.error(f"Error loading image: {e}")

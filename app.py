import streamlit as st
from transformers import pipeline
from PIL import Image
import torch
import io

st.set_page_config(page_title="WeCare Clinical AI", layout="centered")

st.title("🧠 WeCare MedGemma Assistant")
st.write("Upload a medical image and enter your clinical question:")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
prompt = st.text_input("Clinical Question", "Describe this image")

if uploaded_file and prompt:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    pipe = pipeline(
        "image-text-to-text",
        model="google/medgemma-4b-it",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device=0 if torch.cuda.is_available() else -1,
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image", "image": image}]}
    ]

    output = pipe(text=messages, max_new_tokens=200)
    st.markdown("### ✅ AI Response:")
    st.success(output[0]["generated_text"][-1]["content"])

import streamlit as st
import requests
from PIL import Image
import io
import base64

# -- SETUP --
st.set_page_config(page_title="WeCare MedGemma Assistant", layout="centered")
st.title("🧠 WeCare Clinical AI Assistant")
st.write("Upload a clinical image and enter your question. The AI will interpret it.")

# -- INPUTS --
uploaded_file = st.file_uploader("📤 Upload medical image", type=["png", "jpg", "jpeg"])
prompt = st.text_input("💬 Clinical question", "Describe this image")

# -- Hugging Face Inference API Settings --
API_URL = "https://api-inference.huggingface.co/models/google/medgemma-4b-it"
HF_TOKEN = st.secrets["HF_TOKEN"]  # stored securely in Streamlit Cloud Secrets
headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# -- Convert image to base64 (for API payload) --
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# -- Send to Hugging Face API --
def query_medgemma(prompt, image):
    base64_image = encode_image(image)
    payload = {
        "inputs": {
            "past_user_inputs": [],
            "text": prompt,
            "images": [base64_image]
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# -- RUN INFERENCE --
if uploaded_file and prompt:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    with st.spinner("🧠 AI is thinking..."):
        result = query_medgemma(prompt, image)

    st.markdown("### ✅ AI Response:")
    try:
        answer = result[0]["generated_text"] if isinstance(result, list) else result.get("generated_text", "")
        st.success(answer)
    except:
        st.error("⚠️ Error: Could not generate a valid response.")
        st.json(result)

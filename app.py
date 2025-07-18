import streamlit as st
import requests
from PIL import Image
import io
import base64

# ---------------------
# CONFIG
# ---------------------
st.set_page_config(page_title="WeCare MedGemma Assistant", layout="centered")
st.title("🧠 WeCare Clinical AI Assistant")
st.write("Upload a clinical image and enter your question. The AI will interpret it.")

# ---------------------
# USER INPUT
# ---------------------
uploaded_file = st.file_uploader("📤 Upload medical image", type=["png", "jpg", "jpeg"])
prompt = st.text_input("💬 Clinical question", "Describe this image")

# ---------------------
# API SETTINGS
# ---------------------
API_URL = "https://api-inference.huggingface.co/models/google/medgemma-4b-it"
HF_TOKEN = st.secrets["HF_TOKEN"]
headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# ---------------------
# IMAGE TO BASE64
# ---------------------
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ---------------------
# QUERY HUGGING FACE
# ---------------------
def query_medgemma(prompt, image):
    base64_image = encode_image(image)
    payload = {
        "inputs": {
            "past_user_inputs": [],
            "text": prompt,
            "images": [base64_image]
        }
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        st.error(f"❌ HTTP error: {http_err}")
        st.text(f"Response content:\n{response.text}")
        return {"error": "HTTP error"}
    except requests.exceptions.RequestException as req_err:
        st.error(f"❌ Connection error: {req_err}")
        return {"error": "Connection error"}
    except requests.exceptions.JSONDecodeError:
        st.error("❌ Failed to decode JSON — likely the model is still loading or Hugging Face returned an error page.")
        st.text(f"Raw response:\n{response.text}")
        return {"error": "JSON decode error"}

# ---------------------
# RUN INFERENCE
# ---------------------
if uploaded_file and prompt:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded Image", use_column_width=True)

    with st.spinner("🧠 AI is thinking..."):
        result = query_medgemma(prompt, image)

    st.markdown("### ✅ AI Response:")
    try:
        if isinstance(result, list):
            st.success(result[0]["generated_text"])
        elif "generated_text" in result:
            st.success(result["generated_text"])
        else:
            st.warning("⚠️ No valid response received.")
            st.json(result)
    except Exception as e:
        st.error("Unexpected error while parsing response.")
        st.text(str(e))
        st.json(result)

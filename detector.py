import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("./final-model")
    tokenizer = AutoTokenizer.from_pretrained("./final-model")
    return model, tokenizer

st.set_page_config(page_title="AI Text Detector", page_icon="🤖")

st.title("🤖 AI Text Detector")
st.write("Paste any text to check if it's AI-generated or human-written")

# Load model
with st.spinner("Loading model..."):
    model, tokenizer = load_model()

# Text input
text = st.text_area("Enter text to analyze:", height=200, placeholder="Paste your text here...")

if st.button("Analyze Text", type="primary"):
    if text.strip():
        with st.spinner("Analyzing..."):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=1)

            human_prob = predictions[0][0].item() * 100
            ai_prob = predictions[0][1].item() * 100

            # Display results
            st.markdown("---")

            if ai_prob > human_prob:
                st.error(f"🤖 **AI-GENERATED** ({ai_prob:.2f}% confidence)")
            else:
                st.success(f"👤 **HUMAN-WRITTEN** ({human_prob:.2f}% confidence)")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("AI-Generated", f"{ai_prob:.2f}%")
            with col2:
                st.metric("Human-Written", f"{human_prob:.2f}%")
    else:
        st.warning("Please enter some text to analyze")

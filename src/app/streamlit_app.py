import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model
model_path = "fake_real_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
id2label = {0: "Fake", 1: "Real"}

st.title("📰 Fake News Detection App")
st.write("Enter a news headline or article below:")

user_input = st.text_area("News Text")

if st.button("Predict"):
    if user_input.strip():
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()
        st.success(f"**Prediction:** {id2label[pred_label]}  \n**Confidence:** {confidence:.2f}")
    else:
        st.warning("Please enter some text.")
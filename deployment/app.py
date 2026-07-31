import os
import gc
import torch
import numpy as np
import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)

import kagglehub
from transformers import AutoTokenizer

from models.bert_pretrained.arch import SmartMCQPretrained

st.set_page_config(page_title="Smart MCQ Solver", page_icon="🧠")
st.title("Smart MCQ Solver (RoBERTa-Base)")
st.write("Ranked answer prediction using custom fine-tuned RoBERTa model.")

@st.cache_resource
def load_deployment_model():
    """
    Downloads model weights from Kaggle Hub on boot up and caches it.
    This keeps the GitHub repository lightweight while ensuring model availability.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model configuration
    base_model_name = "FacebookAI/roberta-base"
    kaggle_model_handle = "tanmay240/roberta-base/pyTorch/default"
    
    # Download weights from Kaggle Hub
    path = kagglehub.model_download(kaggle_model_handle)
    model_dir = os.path.join(path, "roberta-base")
    
    # Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = SmartMCQPretrained(base_model_name)
    
    weight_path = os.path.join(model_dir, "model.pt")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    
    return tokenizer, model, device

# Boot model
with st.spinner("Downloading weights from Kaggle Hub & Initializing Model..."):
    tokenizer, model, device = load_deployment_model()

# UI
prompt = st.text_area("Question / Prompt", "What is the primary function of an attention mechanism in transformers?")
col1, col2 = st.columns(2)

with col1:
    opt_a = st.text_input("Option A", "Linear scaling of features")
    opt_b = st.text_input("Option B", "Dynamically weighting relevant context tokens")
    opt_c = st.text_input("Option C", "Gradient clipping during backpropagation")

with col2:
    opt_d = st.text_input("Option D", "Dimensionality reduction on embeddings")
    opt_e = st.text_input("Option E", "Data augmentation for raw text")

if st.button("Predict Answers (MAP@3 Ranking)"):
    options = [opt_a, opt_b, opt_c, opt_d, opt_e]
    option_letters = ['A', 'B', 'C', 'D', 'E']
    scores = []
    
    with torch.no_grad():
        for opt_text in options:
            combined_text = f"{prompt} {opt_text}"
            
            encoding = tokenizer(
                combined_text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            ).to(device)
            
            prob = model(encoding['input_ids'], encoding['attention_mask']).item()
            scores.append(prob)
            
    # rank options
    ranked_indices = np.argsort(scores)[::-1]
    top_3_preds = [option_letters[idx] for idx in ranked_indices[:3]]
    prediction_string = " ".join(top_3_preds)
    
    st.success(f"**Top-3 Predicted Ranking:** `{prediction_string}`")
    
    # Probabilities Table
    st.subheader("Raw Prediction Probability Distribution")
    for i in range(5):
        st.write(f"Option **{option_letters[i]}**: `{scores[i]:.4f}`")
        
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

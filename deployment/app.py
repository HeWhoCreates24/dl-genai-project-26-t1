import os
import gc
import sys
import random
import torch
import numpy as np
import streamlit as st

# Path resolution for repository modularity
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)

# Secure Kaggle Authentication
if "KAGGLE_USERNAME" in st.secrets and "KAGGLE_KEY" in st.secrets:
    os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
    os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]

import kagglehub
from transformers import AutoTokenizer
from models.bert_pretrained.arch import SmartMCQPretrained

st.set_page_config(page_title="Smart MCQ Solver", page_icon="🧠", layout="centered")
st.title("🧠 Smart MCQ Solver (RoBERTa-Base)")
st.write("Ranked answer prediction evaluated via MAP@3 using fine-tuned RoBERTa.")

@st.cache_resource
def load_deployment_model():
    """
    Downloads model weights from Kaggle Hub on boot up and caches it.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_model_name = "FacebookAI/roberta-base"
    kaggle_model_handle = "tanmay240/roberta-base/pyTorch/default"
    
    path = kagglehub.model_download(kaggle_model_handle)
    
    # Path fallbacks for Kaggle Hub directory structures
    if os.path.exists(os.path.join(path, "roberta-base", "model.pt")):
        weight_path = os.path.join(path, "roberta-base", "model.pt")
    elif os.path.exists(os.path.join(path, "model.pt")):
        weight_path = os.path.join(path, "model.pt")
    else:
        raise FileNotFoundError("model.pt not found in Kaggle Hub download path.")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = SmartMCQPretrained(base_model_name)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    
    return tokenizer, model, device

# Boot model
with st.spinner("Downloading weights from Kaggle Hub & Initializing Model..."):
    tokenizer, model, device = load_deployment_model()

# Data Dictionary
mcq_data = {
    'id': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
    'prompt': {
        0: "Pick the best possible answer: What is Martin Heidegger's view on the relationship between time and human existence? among the listed options.",
        1: 'What is accelerator-based light-ion fusion?',
        2: 'Determine the correct option: What is the term used in astrophysics to describe light-matter interactions resulting in energy shifts in the radiation field? among the listed options.',
        3: "Select the most accurate option: What is Martin Heidegger's view on the relationship between time and human existence? carefully.",
        4: "Identify the correct statement: What is the concept of simultaneity in Einstein's book, Relativity? carefully."
    },
    'A': {
        0: "Martin Heidegger believes that humans exist within a time continuum that is infinite and does not have a defined beginning or end. The relationship to the past involves acknowledging it as a historical era, and the relationship to the future involves creating a world that will endure beyond one's own time.",
        1: 'Accelerator-based light-ion fusion is a technique that uses particle accelerators to achieve particle kinetic energies sufficient to induce light-ion fusion reactions. This method is relatively easy to implement and can be done in an efficient manner, requiring only a vacuum tube, a pair of electrodes, and a high-voltage transformer. Fusion can be observed with as little as 10 kV between the electrodes.',
        2: 'Blueshifting',
        3: "Martin Heidegger believes that humans exist within a time continuum that is infinite and does not have a defined beginning or end. The relationship to the past involves acknowledging it as a historical era, and the relationship to the future involves creating a world that will endure beyond one's own time.",
        4: 'Simultaneity is relative, meaning that two events that appear simultaneous to an observer in a particular inertial reference frame need not be judged as simultaneous by a second observer in a different inertial frame of reference.'
    },
    'B': {
        0: 'Martin Heidegger believes that humans do not exist inside time, but that they are time. The relationship to the past is a present awareness of having been, and the relationship to the future involves anticipating a potential possibility, task, or engagement.',
        1: 'Accelerator-based light-ion fusion is a technique that uses particle accelerators to achieve particle kinetic energies sufficient to induce heavy-ion fusion reactions. This method is relatively difficult to implement and requires a complex system of vacuum tubes, electrodes, and transformers. Fusion can be observed with as little as 10 kV between the electrodes.',
        2: 'Redshifting',
        3: 'Martin Heidegger believes that humans do not exist inside time, but that they are time. The relationship to the past is a present awareness of having been, and the relationship to the future involves anticipating a potential possibility, task, or engagement.',
        4: 'Simultaneity is relative, meaning that two events that appear simultaneous to an observer in a particular inertial reference frame will always be judged as simultaneous by a second observer in a different inertial frame of reference.'
    },
    'C': {
        0: 'Martin Heidegger does not believe in the existence of time or that it has any effect on human consciousness. The relationship to the past and the future is insignificant, and human existence is solely based on the present.',
        1: 'Accelerator-based light-ion fusion is a technique that uses particle accelerators to achieve particle kinetic energies sufficient to induce light-ion fusion reactions. This method is relatively difficult to implement and requires a complex system of vacuum tubes, electrodes, and transformers. Fusion can be observed with as little as 100 kV between the electrodes.',
        2: 'Reddening',
        3: 'Martin Heidegger does not believe in the existence of time or that it has any effect on human consciousness. The relationship to the past and the future is insignificant, and human existence is solely based on the present.',
        4: 'Simultaneity is absolute, meaning that two events that appear simultaneous to an observer in a particular inertial reference frame will always be judged as simultaneous by a second observer in a different inertial frame of reference.'
    },
    'D': {
        0: 'Martin Heidegger believes that the relationship between time and human existence is cyclical. The past and present are interconnected and the future is predetermined. Human beings do not have free will.',
        1: 'Accelerator-based light-ion fusion is a technique that uses particle accelerators to achieve particle kinetic energies sufficient to induce heavy-ion fusion reactions. This method is relatively easy to implement and can be done in an efficient manner, requiring only a vacuum tube, a pair of electrodes, and a high-voltage transformer. Fusion can be observed with as little as 100 kV between the electrodes.',
        2: 'Whitening',
        3: 'Martin Heidegger believes that the relationship between time and human existence is cyclical. The past and present are interconnected and the future is predetermined. Human beings do not have free will.',
        4: 'Simultaneity is a concept that applies only to Newtonian theories and not to relativistic theories.'
    },
    'E': {
        0: 'Martin Heidegger believes that time is an illusion, and the past, present, and future are all happening simultaneously. Humans exist outside of this illusion and are guided by a higher power.',
        1: 'Accelerator-based light-ion fusion is a technique that uses particle accelerators to achieve particle kinetic energies sufficient to induce light-ion fission reactions. This method is relatively easy to implement and can be done in an efficient manner, requiring only a vacuum tube, a pair of electrodes, and a high-voltage transformer. Fission can be observed with as little as 10 kV between the electrodes.',
        2: 'Yellowing',
        3: 'Martin Heidegger believes that time is an illusion, and the past, present, and future are all happening simultaneously. Humans exist outside of this illusion and are guided by a higher power.',
        4: 'Simultaneity is a concept that applies only to relativistic theories and not to Newtonian theories.'
    },
    'answer': {0: 'B', 1: 'A', 2: 'C', 3: 'B', 4: 'A'}
}

# --- State Management for Randomizer ---
if "current_q_idx" not in st.session_state:
    st.session_state.current_q_idx = 0

if st.button("🎲 Generate Random Question"):
    # Pick a random index between 0 and 4
    st.session_state.current_q_idx = random.randint(0, 4)

idx = st.session_state.current_q_idx

st.info(f"**Loaded Question ID:** {mcq_data['id'][idx]}")

# UI Inputs (Pre-filled with session state data)
prompt = st.text_area("Question / Prompt", mcq_data['prompt'][idx], height=100)
col1, col2 = st.columns(2)

with col1:
    opt_a = st.text_input("Option A", mcq_data['A'][idx])
    opt_b = st.text_input("Option B", mcq_data['B'][idx])
    opt_c = st.text_input("Option C", mcq_data['C'][idx])

with col2:
    opt_d = st.text_input("Option D", mcq_data['D'][idx])
    opt_e = st.text_input("Option E", mcq_data['E'][idx])

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
            
    # Rank options
    ranked_indices = np.argsort(scores)[::-1]
    top_3_preds = [option_letters[idx] for idx in ranked_indices[:3]]
    prediction_string = " ".join(top_3_preds)
    
    st.success(f"**Top-3 Predicted Ranking:** `{prediction_string}`")
    st.info(f"**Ground Truth Answer:** `{mcq_data['answer'][idx]}`")
    
    # Probabilities Table
    st.subheader("Raw Prediction Probability Distribution")
    for i in range(5):
        st.write(f"Option **{option_letters[i]}**: `{scores[i]:.4f}`")
        
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

import os
import gc
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import kagglehub

# setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_df = pd.read_csv('data/test.csv')
sample = pd.read_csv('data/sample_submission.csv')
option_letters = ['A', 'B', 'C', 'D', 'E']

# custom architecture
class SmartMCQPretrained(nn.Module):
    def __init__(self, model_name):
        super(SmartMCQPretrained, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :] 
        x = self.dropout(cls_output)
        x = self.classifier(x)
        return self.sigmoid(x).squeeze(-1)

# inference pipeline
def extract_model_probabilities(model_dir, hf_base_name):
    print(f"\nLoading Tokenizer and Model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = SmartMCQPretrained(hf_base_name).to(device)
    
    # load trained weights
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location=device))
    model.eval()
    
    all_scores = []
    
    with torch.no_grad():
        for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"Predicting {hf_base_name}"):
            prompt = str(row['prompt'])
            question_scores = []
            
            for letter in option_letters:
                option_text = str(row[letter])
                combined_text = f"{prompt} {option_text}"
                
                encoding = tokenizer(
                    combined_text,
                    truncation=True,
                    padding='max_length',
                    max_length=128,
                    return_tensors='pt'
                ).to(device)
                
                prob = model(encoding['input_ids'], encoding['attention_mask']).item()
                question_scores.append(prob)
                
            all_scores.append(question_scores)
            
    # clear vram for next model
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return np.array(all_scores)

print("Downloading models from Kaggle Hub...")
elec_base_path = kagglehub.model_download("tanmay240/electra-base-discriminator/pytorch/default")
bert_base_path = kagglehub.model_download("tanmay240/bert-base-uncased/pytorch/default")
rob_base_path = kagglehub.model_download("tanmay240/roberta-base/pytorch/default")

model_paths = {
    "google/electra-base-discriminator": os.path.join(elec_base_path, "electra-base-discriminator"),
    "google-bert/bert-base-uncased": os.path.join(bert_base_path, "bert-base-uncased"),
    "FacebookAI/roberta-base": os.path.join(rob_base_path, "roberta-base")
}

# extract model probabilities
elec_probs = extract_model_probabilities(model_paths["google/electra-base-discriminator"], "google/electra-base-discriminator")
bert_probs = extract_model_probabilities(model_paths["google-bert/bert-base-uncased"], "google-bert/bert-base-uncased")
rob_probs = extract_model_probabilities(model_paths["FacebookAI/roberta-base"], "FacebookAI/roberta-base")

# hyperparameter weights
w_rob = 0.95
w_elec = 0.04
w_bert = 0.01

print("\nBlending probabilities and ranking answers...")
blended_probs = (w_elec * elec_probs) + (w_bert * bert_probs) + (w_rob * rob_probs)

for i in range(len(test_df)):
    # rank options
    ranked_indices = np.argsort(blended_probs[i])[::-1]
    top_3_preds = [option_letters[idx] for idx in ranked_indices[:3]]
    
    # format submission
    sample.loc[i, 'Prediction'] = " ".join(top_3_preds)

sample.to_csv('src/submission_ensemble.csv', index=False)
print("Saved ensemble submission.csv successfully!")
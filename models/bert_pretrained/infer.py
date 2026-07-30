import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import kagglehub

from arch import SmartMCQPretrained

def run_inference(model_name, max_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_name = model_name.split("/")[-1]
    
    test_csv_path = 'dl-genai-project-26-t1/data/test.csv'
    sample_sub_path = 'dl-genai-project-26-t1/data/sample_submission.csv'

    # download model weights and tokenizer from Kaggle Hub
    path = kagglehub.model_download(f"tanmay240/{model_name}/pyTorch/default")

    model_dir = f"{path}/{model_name}"
    
    test_df = pd.read_csv(test_csv_path)
    sample = pd.read_csv(sample_sub_path)
    option_letters = ['A', 'B', 'C', 'D', 'E']

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = SmartMCQPretrained(model_name)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location=device))
    model.to(device)
    model.eval()

    print(f"Generating test predictions for {clean_name}...")

    with torch.no_grad():
        for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Inference"):
            prompt = str(row['prompt'])
            scores = []
            
            for letter in option_letters:
                option_text = str(row[letter])
                combined_text = f"{prompt} {option_text}"
                
                encoding = tokenizer(
                    combined_text,
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt'
                ).to(device)
                
                prob = model(encoding['input_ids'], encoding['attention_mask']).item()
                scores.append(prob)
                
            # rank predictions
            ranked_indices = np.argsort(scores)[::-1]
            top_3_preds = [option_letters[i] for i in ranked_indices[:3]]
            
            # format submission string
            sample.loc[index, 'Prediction'] = " ".join(top_3_preds)
            
    submission_filename = f'dl-genai-project-26-t1\models\{clean_name}\submission_{clean_name}.csv'
    sample.to_csv(submission_filename, index=False)
    print(f"Saved {submission_filename} successfully!")

if __name__ == "__main__":

    run_inference(
        model_name="google-bert/bert-base-uncased",
        max_length=128
    )
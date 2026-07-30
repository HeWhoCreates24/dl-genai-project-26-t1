import os
import gc
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import kagglehub

from arch import SmartMCQPretrained

def get_probabilities(model_name, test_df, max_length=128):
    """
    Downloads the model from Kaggle Hub, loads it into memory, 
    extracts raw probability scores, and clears GPU memory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_name = model_name.split("/")[-1]

    # Download model weights and tokenizer from Kaggle Hub
    path = kagglehub.model_download(f"tanmay240/{clean_name}/pyTorch/default")
    model_dir = f"{path}/{clean_name}"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = SmartMCQPretrained(model_name)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt"), map_location=device))
    model.to(device)
    model.eval()

    option_letters = ['A', 'B', 'C', 'D', 'E']
    all_probs = []

    print(f"\nExtracting probabilities for {clean_name}...")
    with torch.no_grad():
        for index, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Scoring"):
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
                
            all_probs.append(scores)
            
    # memory cleanup
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return np.array(all_probs)


def run_inference(model_name, max_length=128):
    """
    Loads the test data, fetches probabilities from get_probabilities(), 
    ranks the answers, and formats the final submission CSV.
    """
    clean_name = model_name.split("/")[-1]
    
    test_csv_path = 'data/test.csv'
    sample_sub_path = 'data/sample_submission.csv'
    
    test_df = pd.read_csv(test_csv_path)
    sample = pd.read_csv(sample_sub_path)
    option_letters = ['A', 'B', 'C', 'D', 'E']
    
    # Fetch probabilities
    probs = get_probabilities(model_name, test_df, max_length)
    
    # Rank predictions
    print(f"Ranking answers for {clean_name}...")
    for i in range(len(test_df)):
        ranked_indices = np.argsort(probs[i])[::-1]
        top_3_preds = [option_letters[idx] for idx in ranked_indices[:3]]
        
        # Format submission string
        sample.loc[i, 'Prediction'] = " ".join(top_3_preds)
        
    # Save submission.csv
    output_dir = f'models/{clean_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    submission_filename = os.path.join(output_dir, f'submission_{clean_name}.csv')
    sample.to_csv(submission_filename, index=False)
    print(f"Saved {submission_filename} successfully!")

if __name__ == "__main__":

    run_inference(
        model_name="google-bert/bert-base-uncased",
        max_length=128
    )
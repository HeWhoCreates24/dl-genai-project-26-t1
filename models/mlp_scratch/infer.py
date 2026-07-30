import os
import pickle
import torch
import pandas as pd
import numpy as np

# Import our custom architecture
from arch import SmartMCQSolverMLP

def main():
    test_data_path = 'dl-genai-project-26-t1/data/test.csv'
    sample_sub_path = 'dl-genai-project-26-t1/data/sample_submission.csv'
    model_dir = 'dl-genai-project-26-t1/models/mlp_scratch'

    # Load Data
    test_df = pd.read_csv(test_data_path)
    sample = pd.read_csv(sample_sub_path)
    option_letters = ['A', 'B', 'C', 'D', 'E']

    # Load Vectorizer
    with open(os.path.join(model_dir, "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

    # Load Model Architecture & Weights
    input_dimension = len(vectorizer.vocabulary_)
    model = SmartMCQSolverMLP(input_dimension, hidden_1=512, hidden_2=128)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt")))
    model.eval()

    # Generate Predictions
    print("Generating predictions using the Scratch Model...")
    
    with torch.no_grad():
        for index, row in test_df.iterrows():
            prompt = str(row['prompt'])
            scores = []
            
            for letter in option_letters:
                option_text = str(row[letter])
                combined_text = f"{prompt} {option_text}"
                
                feat = vectorizer.transform([combined_text]).toarray()
                feat_tensor = torch.tensor(feat, dtype=torch.float32)
                
                prob = model(feat_tensor).item()
                scores.append(prob)
                
            # Rank predictions
            ranked_indices = np.argsort(scores)[::-1]
            top_3_preds = [option_letters[i] for i in ranked_indices[:3]]
            
            # Format as submission string
            sample.loc[index, 'Prediction'] = " ".join(top_3_preds)

    # Save submission.csv
    output_path = 'dl-genai-project-26-t1/models/mlp_scratch/submission_scratch.csv'
    sample.to_csv(output_path, index=False)
    print(f"Saved {output_path} successfully!")

if __name__ == "__main__":
    main()
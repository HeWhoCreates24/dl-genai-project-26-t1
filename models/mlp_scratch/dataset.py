import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer

class MCQScratchDataset(Dataset):
    """
    Custom PyTorch Dataset for loading vectorized MCQ text pairs.
    """
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.targets)
        
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def preprocess_training_data(csv_path, max_features=3000):
    """
    Unrolls the MCQs into single classification text pairs and fits a TF-IDF vectorizer.
    """
    train_df = pd.read_csv(csv_path)
    option_letters = ['A', 'B', 'C', 'D', 'E']
    processed_records = []

    for idx, row in train_df.iterrows():
        prompt = str(row['prompt'])
        correct_ans = str(row['answer'])
        
        for letter in option_letters:
            option_text = str(row[letter])
            combined_text = f"{prompt} {option_text}"
            target = 1.0 if letter == correct_ans else 0.0
            
            processed_records.append({
                "text": combined_text,
                "target": target
            })

    processed_df = pd.DataFrame(processed_records)

    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_features = vectorizer.fit_transform(processed_df['text']).toarray()
    y_targets = processed_df['target'].values.astype(np.float32)

    return X_features, y_targets, vectorizer
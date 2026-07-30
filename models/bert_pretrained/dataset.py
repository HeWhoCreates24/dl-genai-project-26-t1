import pandas as pd
import torch
from torch.utils.data import Dataset

class MCQTransformerDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and tokenizing text dynamically.
    """
    def __init__(self, texts, targets, tokenizer, max_length):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        target = self.targets[idx]
        
        # on the fly tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'target': torch.tensor(target, dtype=torch.float32)
        }

def prepare_transformer_data(csv_path):
    """
    Unrolls the MCQs into single classification text pairs.
    """
    df = pd.read_csv(csv_path)
    option_letters = ['A', 'B', 'C', 'D', 'E']
    processed_records = []

    for idx, row in df.iterrows():
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

    return pd.DataFrame(processed_records)
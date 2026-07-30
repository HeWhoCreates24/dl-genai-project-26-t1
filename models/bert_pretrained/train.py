import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

# Import local modules
from arch import SmartMCQPretrained
from dataset import MCQTransformerDataset, prepare_transformer_data

def train_model(model_name, epochs, batch_size, lr, max_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clean_name = model_name.split("/")[-1]
    
    print(f"Preparing training data for {model_name}...")
    train_csv_path = 'data/train.csv'
    processed_df = prepare_transformer_data(train_csv_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = MCQTransformerDataset(
        processed_df['text'].tolist(), 
        processed_df['target'].tolist(), 
        tokenizer, 
        max_length
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SmartMCQPretrained(model_name).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    model.train()
    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(dataloader, desc=f"[{clean_name}] Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * input_ids.size(0)
            binary_preds = (predictions >= 0.5).float()
            correct_predictions += (binary_preds == targets).sum().item()
            total_samples += targets.size(0)
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        
        print(f"\n{clean_name} - Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {epoch_acc:.4f}")

    save_dir = f"models/saved/{clean_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    tokenizer.save_pretrained(save_dir)

    print(f"Saved {clean_name} weights and tokenizer to {save_dir}")

if __name__ == "__main__":

    train_model(
        model_name="google-bert/bert-base-uncased",
        epochs=6,
        batch_size=16,
        lr=2e-5,
        max_length=128
    )
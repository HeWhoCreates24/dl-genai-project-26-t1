import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import our custom modules
from arch import SmartMCQSolverMLP
from dataset import MCQScratchDataset, preprocess_training_data

def main():
    # Configuration
    config = {
            "architecture": "MLP_From_Scratch",
            "hidden_dim_1": 512,
            "hidden_dim_2": 128,
            "learning_rate": 0.001,
            "epochs": 15,
            "batch_size": 32,
            "max_features": 3000
        }

    # Data Preprocessing
    print("Preprocessing data...")

    data_path = 'dl-genai-project-26-t1/data/train.csv'
    X_features, y_targets, vectorizer = preprocess_training_data(data_path, config.max_features)

    # Dataset & DataLoader
    dataset = MCQScratchDataset(X_features, y_targets)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Initialize Model, Loss & Optimizer
    input_dimension = X_features.shape[1]
    model = SmartMCQSolverMLP(input_dimension, config.hidden_dim_1, config.hidden_dim_2)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training Loop
    model.train()
    print("Starting training loop...")

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_features, batch_targets in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_features.size(0)
            binary_preds = (predictions >= 0.5).float()
            correct_predictions += (binary_preds == batch_targets).sum().item()
            total_samples += batch_targets.size(0)
            
        avg_loss = epoch_loss / total_samples
        epoch_acc = correct_predictions / total_samples
        
        print(f"Epoch {epoch+1}/{config.epochs} | Loss: {avg_loss:.4f} | Accuracy: {epoch_acc:.4f}")

    # Save Artifacts
    save_dir = "dl-genai-project-26-t1/models/mlp_scratch"
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    with open(os.path.join(save_dir, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Training finished! Weights and vectorizer saved to {save_dir}")

if __name__ == "__main__":
    main()
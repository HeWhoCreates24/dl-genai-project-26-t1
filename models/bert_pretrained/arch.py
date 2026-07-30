import torch
import torch.nn as nn
from transformers import AutoModel

class SmartMCQPretrained(nn.Module):
    """
    Pretrained Transformer architecture for MCQ solving.
    Uses Hugging Face AutoModel to extract context-aware embeddings.
    """
    def __init__(self, model_name):
        super(SmartMCQPretrained, self).__init__()
        # load pretrained transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        # classifier head
        self.classifier = nn.Linear(self.transformer.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        
        cls_output = outputs.last_hidden_state[:, 0, :] 
        
        x = self.dropout(cls_output)
        x = self.classifier(x)
        return self.sigmoid(x).squeeze(-1)
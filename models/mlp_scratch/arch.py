import torch.nn as nn

class SmartMCQSolverMLP(nn.Module):
    """
    Multilayer Perceptron architecture for MCQ solving built from scratch.
    """
    def __init__(self, input_dim, hidden_1, hidden_2):
        super(SmartMCQSolverMLP, self).__init__()
        
        # Layer 1: Linear transformation followed by a Non-linear activation and Dropout
        self.linear1 = nn.Linear(input_dim, hidden_1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)
        
        # Layer 2: Intermediate layer
        self.linear2 = nn.Linear(hidden_1, hidden_2)
        self.relu2 = nn.ReLU()
        
        # Layer 3: Output layer returning a single probability logit
        self.output_layer = nn.Linear(hidden_2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.dropout1(self.relu1(self.linear1(x)))
        x = self.relu2(self.linear2(x))
        x = self.sigmoid(self.output_layer(x))
        return x
import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self, input_dim=784, out_dim=10):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(input_dim, out_dim)
    
    def forward(self, x):
        # Flatten input for safety
        x = x.view(x.size(0) if len(x.shape) > 1 else 1, -1)
        # Pad or truncate if dimensions do not match exactly
        if x.size(1) > self.fc.in_features:
            x = x[:, :self.fc.in_features]
        elif x.size(1) < self.fc.in_features:
            pad = torch.zeros(x.size(0), self.fc.in_features - x.size(1), dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=1)
        return self.fc(x)

def get_model():
    model = DummyModel()
    model.eval()
    # Dummy forward pass to warm up weights
    with torch.no_grad():
        model(torch.randn(1, 784))
    return model

import torch
import torch.nn as nn
import torch.optim as optim

# Define the multi-task model
class MultiTaskModel(nn.Module):
    # ... (Define the shared encoder and task-specific heads)

# Load and preprocess data for multiple tasks
# ...

# Create the model and optimizer
model = MultiTaskModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # ... (Forward pass for multiple tasks)
        # ... (Calculate loss for each task)
        # ... (Backpropagate and update weights)

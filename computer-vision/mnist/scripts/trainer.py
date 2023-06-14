import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, transforms
import torchvision
from torchsummary import summary
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F



batch_size = 64

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download and load the MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

# Create data loaders
train_loader = t.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = t.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # Conv Layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1) 

        # relu, pool, dropout
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(.25)
        self.dropout2 = nn.Dropout(.5)

        # Dense Layer
        self.fc1 = nn.Linear(7 * 7 * 128, 784)  
        self.fc2 = nn.Linear(784, 10)

    def forward(self, x):
        # Convolution Layer one

        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Convolution Layer Two

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Convolution Layer Three
        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Convolution Layer Four

        x = self.conv4(x)
        x = self.relu(x)
        # x = self.dropout2(x)

        # Convolution Layer Five

        # x = self.conv5(x)
        # x = self.relu(x)
        # x = self.dropout2(x)

        # Flatten 

        x = x.view(x.size(0), -1)

        # Dense Layer One 

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)

        # Dense Layer Two 

        x = self.fc2(x)
        return x

# Create an instance of the model
model = CNNClassifier()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 30
device = t.device("cuda" if t.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

# Test the model
model.eval()
with t.no_grad():
    correct = 0
    total = 0

    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        scores = model(data)
        _, predictions = t.max(scores.data, 1)
        total += targets.size(0)
        correct += (predictions == targets).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    
t.save(model.state_dict(), 'output_model.pt')

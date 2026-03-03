import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split


from dataset import *
from performanceTester import *

batch_size = 512
torch.manual_seed(42)

train_dataset, test_dataset, classes = LoadAnimals10Dataset(batch_size=batch_size, seed=42)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=32,
                            kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=32, out_channels=64,
                            kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=64, out_channels=64,
                            kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 16 * 16, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 10)
        )

    def forward(self, x):
        return self.model(x)

device = 'cpu'

if torch.cuda.is_available():
    print("gpu available")
    device = 'cuda'
else:
    print("gpu not available")

model = CNN().to(device)

for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)

num_epochs = 50
learning_rate = 0.0001
weight_decay = 0.01
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay)


train_loss_list = []
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}:', end=' ')
    train_loss = 0
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss_list.append(train_loss / len(train_loader))
    print(f"Training loss = {train_loss_list[-1]}")

    CalculatePerformanceMetrics(model, test_loader, classes, device, True)

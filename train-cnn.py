import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split


from dataset import *
from performanceTester import *
from aiModels import *

# increase to speed up training (this is limited by vram but also depends on the model)
batch_size = 64

# increase this to nr of logical threads of your machine to speed up training
num_workers = 0
torch.manual_seed(42)

train_dataset, test_dataset, classes = LoadAnimals10Dataset(batch_size=batch_size, seed=42)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)


device = 'cpu'

if torch.cuda.is_available():
    print("gpu available")
    device = 'cuda'
else:
    print("gpu not available")

model = CNN2().to(device)

for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)

num_epochs = 100
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

    if (epoch + 1) % 10 == 0:
        print("Performance Testing:")
        CalculatePerformanceMetrics(model, test_loader, classes, device, True)
        print("Performance Training:")
        CalculatePerformanceMetrics(model, train_loader, classes, device, True)

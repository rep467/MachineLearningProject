import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

data_dir = "./Animals-10"

# load and resize dataset
dataset = ImageFolder(data_dir,transform = transforms.Compose([
    transforms.Resize((128,128)),transforms.ToTensor()
]))

print("Follwing classifications exist: \n",dataset.classes)

total_size = len(dataset)
test_size = int(0.1 * total_size)
train_size = total_size - test_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

batch_size = 1024
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=24)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=24)

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
    

device = 'cuda' if torch.cuda.is_available() else 'cpu'
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


    test_acc = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            y_true = labels.to(device)
            outputs = model(images)
            _, y_pred = torch.max(outputs.data, 1)
            test_acc += (y_pred == y_true).sum().item()

    print(f"Test set accuracy = {100 * test_acc / len(test_dataset)} %")

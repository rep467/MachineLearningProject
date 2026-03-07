import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torchvision.models import vit_b_16, ViT_B_16_Weights

data_dir = "./Animals-10/Animals-10"

weights = ViT_B_16_Weights.DEFAULT
transform = weights.transforms()

dataset = ImageFolder(data_dir, transform=transform)
print("Following classifications exist:\n", dataset.classes)

total_size = len(dataset)
test_size = int(0.1 * total_size)
train_size = total_size - test_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

batch_size = 16
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle = False, num_workers=0)

model = vit_b_16(weights=weights)

num_classes = len(dataset.classes)

in_features = model.heads.head.in_features
model.heads.head = torch.nn.Linear(in_features, num_classes)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)


num_epochs = 50
learning_rate = 0.0001
weight_decay = 0.01
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(   #Adam or AdamW
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

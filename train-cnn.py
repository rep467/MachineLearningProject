import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split


from dataset import *
from train import *
from aiModels import *

# increase to speed up training (this is limited by vram but also depends on the model)
batch_size = 64

# increase this to nr of logical threads of your machine to speed up training
num_workers = 0
torch.manual_seed(42)

# method in dataset.py
train_dataset, test_dataset, val_dataset, classes = LoadAnimals10Dataset(batch_size=batch_size, seed=42)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

# select gpu if available (it should be faster)
# value for amd gpu is also cuda
device = 'cpu'

if torch.cuda.is_available():
    print("gpu available")
    device = 'cuda'
else:
    print("gpu not available")

model = CNN2().to(device)

for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device)

# method in train.py
train(model, val_loader, train_loader, classes, device, 10)
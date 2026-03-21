import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

from dataset import *
from train import *
from aiModels import *

def getDevice():
    if torch.cuda.is_available():
        print("gpu available")
        return 'cuda'
    else:
        print("gpu not available")
        return 'cpu'

def baseScenario(*models):
    torch.manual_seed(42)
    # method in dataset.py
    train_dataset, test_dataset, val_dataset, classes = LoadAnimals10Dataset(seed=42, imageSizeX=224, imageSizeY=224)
    train_loader, test_loader, val_loader = initDataLoaders(train_dataset, test_dataset, val_dataset, num_workers=24, batch_size=16)

    device = 'cpu'
    if torch.cuda.is_available():
        print("gpu available")
        device = 'cuda'
    else:
        print("gpu not available")

    for model in models:
        model = model.to(device)

        train(model, val_loader, train_loader, classes, device, 1, num_epochs=50)

        torch.save(model.state_dict(), f'base_case_{model.getName()}.pth')


baseScenario(ViT(), ViTpretrained(), CNN2())
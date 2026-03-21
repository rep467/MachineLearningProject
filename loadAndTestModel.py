import argparse
import os

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

from dataset import *
from train import *
from aiModels import *
from scenarios import *

def main():
    parser = argparse.ArgumentParser(description="program to load and assess model")

    parser.add_argument("name", type=str, help="model name")
    parser.add_argument("file_path", type=str, help="The location of the model to asses")

    args = parser.parse_args()

    if os.path.exists(args.file_path):
        print("Status: model Found.")
    else:
        print("Status: model NOt Found")

    model = None

    if args.name == 'CNN2':
        model = CNN2()
    elif args.name == 'Vit':
        model = ViT()
    else:
        print("no matching model")
        return
    
    train_dataset, test_dataset, val_dataset, classes = LoadAnimals10Dataset(seed=42, imageSizeX=224, imageSizeY=224)
    test_loader = initDataLoaders(test_dataset, num_workers=24, batch_size=16)
    test_loader = test_loader[0]

    device = getDevice()
    model.to(device)

    model.load_state_dict(torch.load(args.file_path))
    model.eval()

    CalculatePerformanceMetrics(model, test_loader, classes, device, True)


if __name__ == "__main__":
    main()
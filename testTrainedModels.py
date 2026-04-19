from pathlib import Path

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

from dataset import *
from train import *
from aiModels import *
from scenarios import *



train_dataset, test_dataset, val_dataset, classes = LoadAnimals10Dataset(seed=42, imageSizeX=224, imageSizeY=224)
test_loader = initDataLoaders(test_dataset, num_workers=24, batch_size=16)
test_loader = test_loader[0]


device = getDevice()


directory_path = Path('./TrainedModels1')

files = [item for item in directory_path.iterdir() if item.is_file()]

for file in files:
    file = str(file)

    if not file.endswith('.pth'):
        continue

    model = None

    if 'CNN' in file:
        model = CNN2()
    elif 'efficientnet_b1' in file:
        model = EfficientNet()
    else:
        model = ViT()

    model.to(device)
    model.load_state_dict(torch.load(file))
    model.eval()

    result = CalculatePerformanceMetrics(model, test_loader, classes, device, False, True)
    try:
        with open(file[:-3] + 'txt', "w", encoding="utf-8") as fileW:
            fileW.write(result)
    except IOError as e:
        print(f"Error writing to file: {e}")



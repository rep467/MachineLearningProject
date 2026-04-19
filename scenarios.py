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

    device = getDevice()

    trainAndSave(models, 'base_case', val_loader, train_loader, classes, device, 1, num_epochs=50)


def scenarioRotatedAndFlippedImages(*models):
    torch.manual_seed(42)
    # method in dataset.py
    train_dataset, test_dataset, val_dataset, classes = LoadAnimals10Dataset(seed=42, imageSizeX=224, imageSizeY=224)
    train_dataset = rotateAndFlipDataset(train_dataset)
    train_loader, test_loader, val_loader = initDataLoaders(train_dataset, test_dataset, val_dataset, num_workers=24, batch_size=16)
    

    #show_random_batch(train_loader)

    device = getDevice()

    #class_weights = get_class_weights(train_loader, len(classes), device)
    #print(class_weights)

    trainAndSave(models, 'rotated_and_flipped_augmentaition', val_loader, train_loader, classes, device, 1, num_epochs=50)

def classWeights(*models):
    torch.manual_seed(42)
    # method in dataset.py
    train_dataset, test_dataset, val_dataset, classes = LoadAnimals10Dataset(seed=42, imageSizeX=224, imageSizeY=224)
    train_loader, test_loader, val_loader = initDataLoaders(train_dataset, test_dataset, val_dataset, num_workers=24, batch_size=16)
    

    #show_random_batch(train_loader)

    device = 'cpu'
    if torch.cuda.is_available():
        print("gpu available")
        device = 'cuda'
    else:
        print("gpu not available")

    class_weights = get_class_weights(train_loader, len(classes), device)
    #print(class_weights)

    trainAndSave(models, 'class_weights', val_loader, train_loader, classes, device, 1, num_epochs=50, classs_weights=class_weights)

def reducedDataset(*models):
    torch.manual_seed(42)
    # method in dataset.py
    train_dataset, test_dataset, val_dataset, classes = LoadAnimals10DatasetReducedTrainSize(seed=42, imageSizeX=224, imageSizeY=224)
    train_loader, test_loader, val_loader = initDataLoaders(train_dataset, test_dataset, val_dataset, num_workers=24, batch_size=16)

    device = getDevice()

    trainAndSave(models, 'reduced_dataset', val_loader, train_loader, classes, device, 1, num_epochs=50)

def reducedDatasetScenarioRotatedAndFlippedImages(*models):
    torch.manual_seed(42)
    # method in dataset.py
    train_dataset, test_dataset, val_dataset, classes = LoadAnimals10DatasetReducedTrainSize(seed=42, imageSizeX=224, imageSizeY=224)
    train_dataset = rotateAndFlipDataset(train_dataset)
    train_loader, test_loader, val_loader = initDataLoaders(train_dataset, test_dataset, val_dataset, num_workers=24, batch_size=16)
    

    #show_random_batch(train_loader)

    device = getDevice()

    #class_weights = get_class_weights(train_loader, len(classes), device)
    #print(class_weights)

    trainAndSave(models, 'reduced_dataset_rotated_and_flipped_augmentaition', val_loader, train_loader, classes, device, 1, num_epochs=50)

def reducedDatasetClassWeights(*models):
    torch.manual_seed(42)
    # method in dataset.py
    train_dataset, test_dataset, val_dataset, classes = LoadAnimals10DatasetReducedTrainSize(seed=42, imageSizeX=224, imageSizeY=224)
    train_loader, test_loader, val_loader = initDataLoaders(train_dataset, test_dataset, val_dataset, num_workers=24, batch_size=16)
    

    #show_random_batch(train_loader)

    device = getDevice()

    class_weights = get_class_weights(train_loader, len(classes), device)
    #print(class_weights)

    trainAndSave(models, 'reduced_dataset_class_weights', val_loader, train_loader, classes, device, 1, num_epochs=50, classs_weights=class_weights)

if __name__ == "__main__":
    #reducedDataset(EfficientNet(), EfficientNetPretrained())
    #reducedDatasetScenarioRotatedAndFlippedImages(EfficientNet(), EfficientNetPretrained())
    #reducedDatasetClassWeights(EfficientNet(), EfficientNetPretrained())
    #classWeights(EfficientNet(), EfficientNetPretrained())
    #scenarioRotatedAndFlippedImages(EfficientNet(), EfficientNetPretrained())
    #pass

    reducedDataset(CNN2(), ViT(), ViTpretrained(), EfficientNet(), EfficientNetPretrained())
    reducedDatasetScenarioRotatedAndFlippedImages(CNN2(), ViT(), ViTpretrained(), EfficientNet(), EfficientNetPretrained())
    reducedDatasetClassWeights(CNN2(), ViT(), ViTpretrained(), EfficientNet(), EfficientNetPretrained())
    classWeights(CNN2(), ViT(), ViTpretrained(), EfficientNet(), EfficientNetPretrained())
    baseScenario(ViTpretrained(), ViT(), CNN2(), EfficientNet(), EfficientNetPretrained())
    scenarioRotatedAndFlippedImages(CNN2(), ViT(), ViTpretrained(), EfficientNet(), EfficientNetPretrained())


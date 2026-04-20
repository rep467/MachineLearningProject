#Author: Per Sander, Dominic Smith, Alexander Go
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

from dataset import *
from train import *
from aiModels import *

# helper function to get device string
def getDevice():
    if torch.cuda.is_available():
        print("gpu available")
        return 'cuda'
    else:
        print("gpu not available")
        return 'cpu'
'''
Base scenario
All other scenarios are based on this with minor changes which defines there unique scenario

Setup of the scenario is
* get dataset using seed for consistent split of data
* get device
* call train for all models inputed
'''
def baseScenario(*models):
    torch.manual_seed(42)
    # method in dataset.py
    train_dataset, test_dataset, val_dataset, classes = LoadAnimals10Dataset(seed=42, imageSizeX=224, imageSizeY=224)
    train_loader, test_loader, val_loader = initDataLoaders(train_dataset, test_dataset, val_dataset, num_workers=24, batch_size=16)

    device = getDevice()

    trainAndSave(models, 'base_case', val_loader, train_loader, classes, device, 1, num_epochs=50)

'''
Scenario with randomly rotated images 
function for random rotation is defined in dataset.py
'''
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

'''
Scenario which parses the class weights to the training method
This results in the training algorithm applying class weights during training
'''
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

'''
Scenario which is the base case for the reduced training dataset

function for selecting the reduced training set is defined in dataset.py
'''
def reducedDataset(*models):
    torch.manual_seed(42)
    # method in dataset.py
    train_dataset, test_dataset, val_dataset, classes = LoadAnimals10DatasetReducedTrainSize(seed=42, imageSizeX=224, imageSizeY=224)
    train_loader, test_loader, val_loader = initDataLoaders(train_dataset, test_dataset, val_dataset, num_workers=24, batch_size=16)

    device = getDevice()

    trainAndSave(models, 'reduced_dataset', val_loader, train_loader, classes, device, 1, num_epochs=50)

'''
Scenario which combines the reduced training dataset and the rotated and flipped scenario

function for selecting the reduced training set is defined in dataset.py

function for random rotation is defined in dataset.py
'''
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
    
'''
Scenario which combines the reduced training dataset and the class weights scenario

function for selecting the reduced training set is defined in dataset.py

class weights are parsed to training function
'''
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

'''
main function to run the entire experiment. 

This will call every scenario defined in this file with 5 different models (custom CNN (CNN2), vit_16_b (Vit), vit_16_b pretrained (ViTpretrained), EfficientNet, EfficientNet pretrained (EfficientNetPretrained))
The models are defined in aiModels.py
'''
if __name__ == "__main__":
    reducedDataset(CNN2(), ViT(), ViTpretrained(), EfficientNet(), EfficientNetPretrained())
    reducedDatasetScenarioRotatedAndFlippedImages(CNN2(), ViT(), ViTpretrained(), EfficientNet(), EfficientNetPretrained())
    reducedDatasetClassWeights(CNN2(), ViT(), ViTpretrained(), EfficientNet(), EfficientNetPretrained())
    classWeights(CNN2(), ViT(), ViTpretrained(), EfficientNet(), EfficientNetPretrained())
    baseScenario(ViTpretrained(), ViT(), CNN2(), EfficientNet(), EfficientNetPretrained())
    scenarioRotatedAndFlippedImages(CNN2(), ViT(), ViTpretrained(), EfficientNet(), EfficientNetPretrained())


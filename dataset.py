import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

from torch.utils.data import Dataset
from torchvision import transforms

data_dir = "./Animals-10/Animals-10"

class TransformedSubset(Dataset):
    def __init__(self, set, transform=None):
        self.set = set
        self.transform = transform

    def __getitem__(self, i):
        x, y = self.set[i]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.set)

def rotateAndFlipDataset(dataset):
    train_augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15)
    ])

    dataset = TransformedSubset(dataset)
    dataset.transform = train_augmentation

    return dataset





def LoadAnimals10Dataset(imageSizeX = 128, imageSizeY = 128, printDebugClasses = False, seed = None, batch_size = 64, num_workers=0):
    dataset = ImageFolder(data_dir,transform = transforms.Compose([
        transforms.Resize((imageSizeX,imageSizeY)),transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))

    if printDebugClasses:
        print("Follwing classifications exist: \n",dataset.classes)

    total_size = len(dataset)
    test_size = int(0.1 * total_size)
    val_size = test_size
    train_size = total_size - 2*test_size

    gen = torch.Generator()

    if seed != None:
        gen.manual_seed(seed)

    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size], generator=gen)
    return train_dataset, test_dataset, val_dataset, dataset.classes

def initDataLoaders(*datasets, num_workers, batch_size):
    return tuple(torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True) for dataset in datasets)
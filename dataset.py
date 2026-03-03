import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

data_dir = "./Animals-10/Animals-10"


def LoadAnimals10Dataset(imageSizeX = 128, imageSizeY = 128, printDebugClasses = False, seed = None, batch_size = 64, num_workers=0):
    dataset = ImageFolder(data_dir,transform = transforms.Compose([
        transforms.Resize((imageSizeX,imageSizeY)),transforms.ToTensor()
    ]))

    if printDebugClasses:
        print("Follwing classifications exist: \n",dataset.classes)

    total_size = len(dataset)
    test_size = int(0.1 * total_size)
    train_size = total_size - test_size

    gen = torch.Generator()

    if seed != None:
        gen.manual_seed(seed)

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=gen)

    return train_dataset, test_dataset, dataset.classes

import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights

class CNN1(torch.nn.Module):
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
    
    def getName(self):
        return "CNN-custom-1"


class CNN2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),

                # Block 2
                torch.nn.Conv2d(64, 128, 3, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),

                # Block 3
                torch.nn.Conv2d(128, 256, 3, padding=1),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),

                # Block 4
                torch.nn.Conv2d(256, 512, 3, padding=1),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((1, 1)),

                torch.nn.Flatten(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(512, 512),

                torch.nn.ReLU(),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.model(x)
        
    def getName(self):
        return "CNN-custom-2"

class ViT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = vit_b_16(weights=None)

        in_features = self.model.heads.head.in_features
        self.model.heads.head = torch.nn.Linear(in_features, 10)

    def forward(self, x):
        return self.model(x)
    
    def getName(self):
        return "vit_b_16"

class ViTpretrained(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        in_features = self.model.heads.head.in_features
        self.model.heads.head = torch.nn.Linear(in_features, 10)

    def forward(self, x):
        return self.model(x)
    
    def getName(self):
        return "vit_b_16_pretrained"

import torch, torchvision, matplotlib as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


#Define a standard size




# Download training data from open datasets.
training_data = torchvision.datasets.OxfordIIITPet(
    root="data",
    split='trainval',
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = torchvision.datasets.OxfordIIITPet(
    root="data",
    split='test',
    download=True,
    transform=ToTensor(),
)


for img, label in training_data:
    print(img.shape)
import torch, torchvision
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.optim as optim
import torch.nn.functional as F
from train import my_transform, NeuralNetwork




# Download training data from open datasets and preprocess images
training_data = torchvision.datasets.OxfordIIITPet(
    root="data",
    split='trainval',
    download=True,
    transform=my_transform
)

# Download test data from open datasets.

test_data = torchvision.datasets.OxfordIIITPet(
    root="data",
    split='test',
    download=True,
    transform=my_transform,
)

#Loaders
train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

# Testing
#Load the model
tested_model = NeuralNetwork()
tested_model.load_state_dict(torch.load('model.pth'))




# Set evaluation mode and disable gradient
tested_model.eval()
with torch.no_grad():
    total_correct = 0
    # Make Predictions


    #Check for every input for the training data_set 

    #Training dataset accuracy
    for inputs, labels in train_dataloader:
        predictions = tested_model(inputs) 

        if torch.argmax(predictions) == labels:
            total_correct += 1

    training_set_accuracy = round(total_correct/training_data.__len__()*100,2)
    total_correct = 0


    #Test dataset accuracy
    for inputs, labels in test_dataloader:
        predictions = tested_model(inputs) 

        if torch.argmax(predictions) == labels:
            total_correct += 1

    test_set_accuracy  = round(total_correct/test_data.__len__()*100,2)


    print(f"Accuracy for training dataset {training_set_accuracy}%")
    print(f"Accuracy for test dataset {test_set_accuracy}%")





# Print the test accuracy and loss


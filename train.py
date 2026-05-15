import torch, torchvision
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.optim as optim
import torch.nn.functional as F
import PIL

# Load Data

#Augment the data images to useful size and form
my_transform = torchvision.transforms.Compose([
    #Resize image
    torchvision.transforms.Resize((224,224)),

    # Change output channel from 3(RGB COLOURS) to 1(Greyscale) (Disabled)
    # torchvision.transforms.Grayscale(num_output_channels = 1),
    PIL.ImageOps.autocontrast(),

    

    #Change to tensor
    torchvision.transforms.ToTensor(),


    #Normalise image so that tensors and neural network works better on them, recommended values from https://www.geeksforgeeks.org/python/how-to-normalize-images-in-pytorch/
    torchvision.transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]) #mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
])




# Download training data from open datasets and preprocess images
training_data = torchvision.datasets.OxfordIIITPet(
    root="data",
    split='trainval',
    download=True,
    transform=my_transform
)

# Download test data from open datasets.
"""
test_data = torchvision.datasets.OxfordIIITPet(
    root="data",
    split='test',
    download=True,
    transform=ToTensor(),
)
"""

# Working labels map for convenience, looked at given txt file of the dataset.
labels_map = {
    0: "Abyssinian",
    1: "American Bulldog",
    2: "American Pit Bull Terrier",
    3: "Basset Hound",
    4: "Beagle",
    5: "Bengal",
    6: "Birman",
    7: "Bombay",
    8: "Boxer",
    9: "British Shorthair",
    10: "Chihuahua",
    11: "Egyptian Mau",
    12: "English Cocker Spaniel",
    13: "English Setter",
    14: "German Shorthaired",
    15: "Great Pyrenees",
    16: "Havanese",
    17: "Japanese Chin",
    18: "Keeshond",
    19: "Leonberger",
    20: "Maine Coon",
    21: "Minature Pinscher",
    22: "Newfoundland",
    23: "Persian",
    24: "Pomeranian",
    25: "Pug",
    26: "Ragdoll",
    27: "Russian Blue",
    28: "Saint Bernard",
    29: "Samoyed",
    30: "Scottish Terrier",
    31: "Shiba Inu",
    32: "Siamese",
    33: "Sphynx",
    34: "Staffordshire Bull Terrier",
    35: "Wheaten Terrier",
    36: "Yorkshire Terrier",
}

train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

print(device)



# Display image and label.

train_features, train_labels = next(iter(train_dataloader))

print(train_features.size())
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
print(img.size())

# Swap axes for channel to work
img = img.swapaxes(0,1)
img = img.swapaxes(1,2)
print(img)
print(f"Label: {label} which is a", labels_map[label.item()])
# Gives black and white image (img[:,:,2]) if channel is 3
plt.imshow(img, cmap='grey')
plt.show()



class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 12, kernel_size= 11) # Input img size+1 - kernel size 
        self.conv2 = nn.Conv2d(in_channels=12, out_channels= 24, kernel_size= 8) # (128, 214, 214)

        self.pool = nn.MaxPool2d(2, stride=2) # -->(128, 107, 107)

        self.inputlayer = nn.Linear(24 * 50 * 50, 120)
        self.fc2 = nn.Linear(120, 84)
        self.outlayer = nn.Linear(84, len(labels_map))


    def forward(self, x): # x.size() = (3, 224, 224)
        x = F.relu(self.conv1(x)) #-> (12, 214, 124)
        x = self.pool(x) #-> (12, 107, 107)
        x = F.relu(self.conv2(x)) #-> (24, 100, 100 )
        x = self.pool(x) # -> (24, 50, 50)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.inputlayer(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.outlayer(x), dim=1)
        return x


network = NeuralNetwork()
network.to(device)

loss_function = nn.NLLLoss()
optimizer = optim.Adam(network.parameters(), lr=0.001)

epochs = 0
for epoch in range(epochs):
    for images, labels in train_dataloader:
        optimizer.zero_grad()

        output = network(images)
        loss = loss_function(output, labels)
        
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


print('Finished Training')
import torch, torchvision
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

# Load Data

#Augment the data images to useful size and form
my_transform = torchvision.transforms.Compose([
    #Resize image
    torchvision.transforms.Resize((224,224)),

    # Change output channel from 3(RGB COLOURS) to 1(Greyscale) (Disabled)
    # torchvision.transforms.Grayscale(num_output_channels = 1),
    

    #Change to tensor
    torchvision.transforms.ToTensor(),


    #Normalise image so that tensors and neural network works better on them, recommended values from https://www.geeksforgeeks.org/python/how-to-normalize-images-in-pytorch/
    torchvision.transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) #mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
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




# Display image and label.
"""
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
"""




# Defining the network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutions layer change the channel or amound of features from the image.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 32, kernel_size= 3, padding=1) 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels= 64, kernel_size= 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels= 128, kernel_size= 3, padding=1)

        # Take max value of a 2x2 grid of pixels, helps performance
        self.pool = nn.MaxPool2d(2, stride=2)

        self.inputlayer = nn.Linear(128 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 45)
        self.outlayer = nn.Linear(45, len(labels_map))


    def forward(self, x): # x.size() = (3, 224, 224)
        x = F.relu(self.conv1(x)) #-> (32, 224, 224)
        x = self.pool(x) #-> (32, 112, 112)

        x = F.relu(self.conv2(x)) #-> (64, 112, 112 )
        x = self.pool(x) # -> (64, 56, 56)

        x = F.relu(self.conv3(x)) #-> (128, 56, 56 )
        x = self.pool(x) #-> (128, 28, 28)


        x = torch.flatten(x, 1) # flatten all dimensions except batch

        x = F.relu(self.inputlayer(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.outlayer(x), dim=1)
        return x



# Load data and check device

def main():

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    print(device)


    #Training
    network = NeuralNetwork()
    network.to(device)

    # loss function opitmized for log_softmax function
    loss_function = nn.NLLLoss()

    #The optimizer changes the weights
    optimizer = optim.Adam(network.parameters(), lr=0.001)

    # Epoch (lap around whole dataset)
    epochs = 14
    print("Training...")
    for epoch in range(epochs):
        for images, labels in train_dataloader:
            optimizer.zero_grad()

            output = network(images)

            # Calculate loss
            loss = loss_function(output, labels)
            
            # Backward propagation
            loss.backward()

            #Change the weights
            optimizer.step()


        #Loss for every epoch
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


    print('Finished Training')


    # Save the model
    torch.save(network.state_dict(), 'model15.pth')



if __name__ == '__main__':
    main()
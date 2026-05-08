import torch, torchvision
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Load Data

#Augment the data images to useful size and form
my_transform = torchvision.transforms.Compose([
    #Resize image
    torchvision.transforms.Resize((224,224)),

    # Change output channel from 3(RGB COLOURS) to 1(Greyscale)
    torchvision.transforms.Grayscale(num_output_channels = 1),

    #Change to tensor
    torchvision.transforms.ToTensor()
])




# Download training data from open datasets.
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

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

print("hello")

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
print(img.size())

# Swap axes for channel to work
#img = img.swapaxes(0,1)
#img = img.swapaxes(1,2)

print(f"Label: {label} which is a", labels_map[label.item()])
# Gives black and white image (img[:,:,2]) if channel is 3
print(img)
plt.imshow(img, cmap='grey')
plt.show()



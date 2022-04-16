from import_package import *

'''
Dataset, Data Loader, and Transforms
Torchvision provides lots of useful utilities for image preprocessing, data wrapping as well as data augmentation.
Here, since our data are stored in folders by class labels, we can directly apply torchvision.datasets.DatasetFolder for wrapping data without much effort.
Please refer to PyTorch official website for details about different transforms.
'''
# It is important to do data augmentation in training.
# However, not every augmentation is useful.
# Please think about what kind of augmentation is helpful for food recognition.
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((256, 256)),
    # You may add some transforms here.
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomResizedCrop(128),
    # transforms.RandomRotation(degrees=(-45, 45), fill=0),
    # transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.1, 0.1)),
    # transforms.RandomGrayscale(),
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
    # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    # transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])


# Construct datasets.
class IrisTrainingDataset(Dataset):
    def __init__(self, csv_path: str, image_folder: str, transform=None):
        self.csv_label = pd.read_csv(csv_path)
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self):
        return len(self.csv_label)

    def __getitem__(self, item):
        filename = self.csv_label.loc[item].at["filename"]
        label = self.csv_label.loc[item].at["category"]
        image = Image.open(os.path.join(self.image_folder, filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, label

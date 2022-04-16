from import_package import *

'''
Model
The basic model here is simply a stack of convolutional layers followed by some fully-connected layers.
Since there are three channels for a color image (RGB), the input channels of the network must be three. In each convolutional layer, typically the channels of inputs grow, while the height and width shrink (or remain unchanged, according to some hyperparameters like stride and padding).
Before fed into fully-connected layers, the feature map must be flattened into a single one-dimensional vector (for each image). These features are then transformed by the fully-connected layers, and finally, we obtain the "logits" for each class.

WARNING -- You Must Know
You are free to modify the model architecture here for further improvement. However, if you want to use some well-known architectures such as ResNet50, please make sure NOT to load the pre-trained weights. Using such pre-trained models is considered cheating and therefore you will be punished. Similarly, it is your responsibility to make sure no pre-trained weights are used if you use torch.hub to load any modules.
For example, if you use ResNet-18 as your model:
model = torchvision.models.resnet18(pretrained=False) → This is fine.
model = torchvision.models.resnet18(pretrained=True) → This is NOT allowed.
'''


class IrisClassifier(nn.Module):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [#, 3, 640, 480] (#: Number of inputs)
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # [#, 64, 640, 480]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [#, 64, 320, 240]

            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1),  # [#, 256, 320, 240]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),  # [#, 256, 160, 120]

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # [#, 512, 160, 120]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.MaxPool2d(16, 16, 0),  # [#, 512, 20, 15]
        )
        self.fc_layers = nn.Sequential(  # fc = fully-connected
            nn.Linear(512 * 4 * 4, 4096),  # [#, 4096]
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(1024, 219)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 640, 480]
        # output: [batch_size, 219]

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)

        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)  # [#, 2457600]

        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x

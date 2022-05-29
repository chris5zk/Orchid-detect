from dataset import *
from model import *
from train import *

total_dataset = IrisTrainingDataset(label_path, training_path, train_tfm)
train_size = int(len(total_dataset) * train_ratio)
valid_size = len(total_dataset) - train_size
train_set, valid_set = random_split(total_dataset, [train_size, valid_size])

# Construct data loaders.
print(f"{Bcolors.WARNING}Creating train_loader...{Bcolors.ENDC}", file=sys.stderr)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
print(f"{Bcolors.WARNING}Creating valid_loader...{Bcolors.ENDC}", file=sys.stderr)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

# "cuda" only when GPUs are available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize a model, and put it on the device specified.
print(f"{Bcolors.WARNING}Putting model into {device}...{Bcolors.ENDC}", file=sys.stderr)
# model = IrisClassifier().to(device)

# model = googlenet(pretrained=True)  # This model will download to "$HOME/.cache/torch/hub/checkpoints/"
# model.fc = nn.Sequential(
#     model.fc,
#     nn.ReLU(inplace=True),
#     nn.Linear(in_features=1000, out_features=219, bias=True)
# )
# model = model.to(device)
# print("------------------------- ↓ GoogleNet ↓ -------------------------")
# print(model)
# print("------------------------- ↑ GoogleNet ↑ -------------------------", flush=True)

model = resnext50_32x4d(pretrained=True)
model.fc = nn.Sequential(
    model.fc,
    nn.ReLU(inplace=True),
    nn.Linear(in_features=1000, out_features=219, bias=True)
)
model = model.to(device)
print("------------------------- ↓ ResNeXt50 ↓ -------------------------")
print(model)
print("------------------------- ↑ ResNeXt50 ↑ -------------------------", flush=True)
# quit(0)

# For the classification task, we use cross-entropy as the measurement of performance.
# nn.CrossEntropyLoss includes softmax function
criterion = nn.CrossEntropyLoss()
# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# Train model
train(model, criterion, device, train_loader, valid_loader, optimizer)

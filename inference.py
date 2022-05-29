from hyperparams import *
from model import *
from import_package import *

'''
Testing
For inference, we need to make sure the model is in eval mode, and the order of the dataset should not be shuffled ("shuffle=False" in test_loader).
Last but not least, don't forget to save the predictions into a single CSV file. The format of CSV file should follow the rules mentioned in the slides.
'''

# "cuda" only when GPUs are available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_set = None  # We don't have test dataset yet.
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# model = IrisClassifier().to(device)

# model = googlenet(pretrained=True)
# model.fc = nn.Sequential(
#     model.fc,
#     nn.ReLU(inplace=True),
#     nn.Linear(in_features=1000, out_features=219, bias=True)
# )
# model = model.to(device)

model = resnext50_32x4d(pretrained=True)
model.fc = nn.Sequential(
    model.fc,
    nn.ReLU(inplace=True),
    nn.Linear(in_features=1000, out_features=219, bias=True)
)
model = model.to(device)

# Make sure the model is in eval mode.
# Some modules like Dropout or BatchNorm affect if the model is in training mode.
if os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path))
else:
    print(f"{Bcolors.FAIL}Error, model {model_path} does not exist. Exiting.", file=sys.stderr)
    quit(1)

model.eval()

# Initialize a list to store the predictions.
predictions = []

# Iterate the testing set by batches.
for batch in tqdm(test_loader):
    # A batch consists of image data and corresponding labels.
    # But here the variable "labels" is useless since we do not have the ground-truth.
    # If printing out the labels, you will find that it is always 0.
    # This is because the wrapper (DatasetFolder) returns images and labels for each batch,
    # so we have to create fake labels to make it work normally.
    imgs, labels = batch
    imgs = imgs.to(device)

    # We don't need gradient in testing, and we don't even have labels to compute loss.
    # Using torch.no_grad() accelerates the forward process.
    with torch.no_grad():
        logits = model(imgs)

    # Take the class with greatest logit as prediction and record it.
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

# Save predictions into the file.
with open("predict.csv", "w") as f:
    # The first row must be "Id, Category"
    f.write("Id,Category\n")

    # For the rest of the rows, each image id corresponds to a predicted class.
    for i, pred in enumerate(predictions):
        f.write(f"{i},{pred}\n")
    print("The prediction has been written successfully.")

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
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Initialize a model, and put it on the device specified.
print(f"{Bcolors.WARNING}Putting model into cuda...{Bcolors.ENDC}", file=sys.stderr)
model = IrisClassifier().to(device)
model.device = device

# For the classification task, we use cross-entropy as the measurement of performance.
# nn.CrossEntropyLoss includes softmax function
criterion = nn.CrossEntropyLoss()
# Train model
train(model, criterion, train_loader, valid_loader)

'''
Testing
For inference, we need to make sure the model is in eval mode, and the order of the dataset should not be shuffled ("shuffle=False" in test_loader).
Last but not least, don't forget to save the predictions into a single CSV file. The format of CSV file should follow the rules mentioned in the slides.

WARNING -- Keep in Mind
Cheating includes but not limited to:
1. using testing labels,
2. submitting results to previous Kaggle competitions,
3. sharing predictions with others,
4. copying codes from any creatures on Earth,
5. asking other people to do it for you.
Any violations bring you punishments from getting a discount on the final grade to failing the course.

It is your responsibility to check whether your code violates the rules. When citing codes from the Internet, you should know what these codes exactly do. You will NOT be tolerated if you break the rule and claim you don't know what these codes do.
'''

# We don't have test dataset yet.
'''
# Make sure the model is in eval mode.
# Some modules like Dropout or BatchNorm affect if the model is in training mode.
model.load_state_dict(torch.load(model_path))
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

    # We don't need gradient in testing, and we don't even have labels to compute loss.
    # Using torch.no_grad() accelerates the forward process.
    with torch.no_grad():
        logits = model(imgs.to(device))

    # Take the class with greatest logit as prediction and record it.
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

# Save predictions into the file.
with open("predict.csv", "w") as f:
    # The first row must be "Id, Category"
    print('Using model with best validation loss {:.5f} and accuracy {:.5f} to make prediction.'.format(best_loss, best_acc))
    f.write("Id,Category\n")

    # For the rest of the rows, each image id corresponds to a predicted class.
    for i, pred in enumerate(predictions):
        f.write(f"{i},{pred}\n")
    print("The prediction has been written successfully.")
'''

from import_package import *

dataset_path = "./Iris_Dataset"
training_path = f"{dataset_path}/Dataset"
label_path = f"{dataset_path}/label.csv"
model_path = "./model.ckpt"
tensorboard_log_path = "./runs"

# Batch size for training, validation, and testing.
# A greater batch size usually gives a more stable gradient.
# But the GPU memory is limited, so please adjust it carefully.
batch_size = 128
train_ratio = 0.7
# The number of training epochs.
early_stop = 1000
n_epochs = 10000  # n_epochs must greater than 1
learning_rate = 1e-3
weight_decay = 1e-5

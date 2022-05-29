from hyperparams import *


def train(model, criterion, device, train_loader, valid_loader, optimizer):
    early_stop_count = 0
    best_loss = 2147483647  # INT_MAX
    if not os.path.exists(tensorboard_log_path):
        os.mkdir(tensorboard_log_path)
    elif len(os.listdir(tensorboard_log_path)) != 0:
        print(f"{tensorboard_log_path} exists logs, do you want to clean logs? (y/n)")
        yn = input()
        if yn == "y" or yn == "Y":
            print("Clearing logs.")
            os.system(f"rm {tensorboard_log_path}/* -f")
        else:
            print("Keep logs.")
    writer = SummaryWriter(log_dir=tensorboard_log_path)  # Writer of tensorboard.
    if os.path.isfile(model_path):
        print(f"{model_path} exists, do you want to load the last model? (y/n)")
        yn = input()
        if yn == "y" or yn == "Y":
            print(f"{Bcolors.WARNING}Loading last model{Bcolors.ENDC}", file=sys.stderr)
            model.load_state_dict(torch.load(model_path), strict=False)
    try:
        for epoch in range(n_epochs):
            # ---------- Training ----------
            # Make sure the model is in train mode before training.
            model.train()

            # These are used to record information in training.
            train_loss = []
            train_accs = []

            # Iterate the training set by batches.
            for batch in tqdm(train_loader):
                # A batch consists of image data and corresponding labels.
                imgs, labels = batch
                imgs = imgs.to(device)
                labels = labels.to(device)

                # Forward the data. (Make sure data and model are on the same device.)
                logits = model(imgs)

                # Calculate the cross-entropy loss.
                # We don't need to apply softmax before computing cross-entropy as it is done automatically.
                loss = criterion(logits, labels)

                # Gradients stored in the parameters in the previous step should be cleared out first.
                optimizer.zero_grad()

                # Compute the gradients for parameters.
                loss.backward()

                # Clip the gradient norms for stable training.
                # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

                # Update the parameters with computed gradients.
                optimizer.step()

                # Compute the accuracy for current batch.
                acc = (logits.argmax(dim=-1) == labels.to(device)).double().mean()

                # Record the loss and accuracy.
                train_loss.append(loss.detach().item())  # add detach()
                train_accs.append(acc)

            # The average loss and accuracy of the training set is the average of the recorded values.
            train_loss = sum(train_loss) / len(train_loss)
            train_acc = sum(train_accs) / len(train_accs)

            # Print the information.
            print(f"{Bcolors.OKBLUE}[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}{Bcolors.ENDC}", file=sys.stderr)

            # ---------- Validation ----------
            # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
            model.eval()

            # These are used to record information in validation.
            valid_loss = []
            valid_accs = []

            # Iterate the validation set by batches.
            for batch in tqdm(valid_loader):
                # A batch consists of image data and corresponding labels.
                imgs, labels = batch

                # We don't need gradient in validation.
                # Using torch.no_grad() accelerates the forward process.
                with torch.no_grad():
                    logits = model(imgs.to(device))

                # We can still compute the loss (but not the gradient).
                loss = criterion(logits, labels.to(device))

                # Compute the accuracy for current batch.
                acc = (logits.argmax(dim=-1) == labels.to(device)).double().mean()

                # Record the loss and accuracy.
                valid_loss.append(loss.detach().item())  # add detach()
                valid_accs.append(acc)

            # The average loss and accuracy for entire validation set is the average of the recorded values.
            valid_loss = sum(valid_loss) / len(valid_loss)
            valid_acc = sum(valid_accs) / len(valid_accs)

            # Print the information.
            print(f"{Bcolors.OKCYAN}[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}{Bcolors.ENDC}", file=sys.stderr)

            # Write result to tensorboard writer
            writer.add_scalar("Train/Step", train_loss, epoch)
            writer.add_scalar("Valid/Step", valid_loss, epoch)
            # if the model improves, save a checkpoint at this epoch
            if epoch == 0 or best_loss > valid_loss:
                best_loss = valid_loss
                best_acc = valid_acc
                print(f"{Bcolors.WARNING}Saving model with validation loss {best_loss:.5f} and accuracy {best_acc:.5f}{Bcolors.ENDC}", file=sys.stderr)
                torch.save(model.state_dict(), model_path)
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count >= early_stop:
                print(f"{Bcolors.FAIL}Our model is not improving with {early_stop_count} steps. Stop.{Bcolors.ENDC}", file=sys.stderr)
                break
    except KeyboardInterrupt:
        print(f"{Bcolors.FAIL}stop!!!{Bcolors.ENDC}", file=sys.stderr)

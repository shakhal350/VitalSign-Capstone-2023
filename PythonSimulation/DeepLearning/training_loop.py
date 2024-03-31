import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from mvn_model_def import ViTForHeartRateEstimation, MAEModel, freeze_parameters
from data_preparation import load_and_preprocess_multiple_files, create_datasets
import matplotlib.pyplot as plt
import pandas as pd




def train_vit(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience=20):
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    epoch_losses = []  # Store average training loss per epoch
    val_epoch_losses = []  # Store average validation loss per epoch

    for epoch in range(epochs):
        running_loss = 0.0
        batch_count = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

        avg_epoch_loss = running_loss / batch_count
        epoch_losses.append(avg_epoch_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                labels = None
                if len(batch) == 2:  # Handles case where labels are provided
                    labels = batch[1].to(device)

                outputs = model(inputs)
                if labels is not None:  # Compute loss only if labels are available
                    v_loss = criterion(outputs, labels.unsqueeze(1))
                    val_loss += v_loss.item()
            
        avg_val_loss = val_loss / len(val_loader) if labels is not None else 0
        val_epoch_losses.append(avg_val_loss)

        print(f'Epoch {epoch+1}: Training Loss = {avg_epoch_loss:.4f}, Validation Loss = {avg_val_loss:.4f}')

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # Reset patience
            # Save model checkpoint
            torch.save(model.state_dict(), 'vit_heart_rate_model_best_newest.pth')
            print("Validation loss decreased. Saving model...")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

        model.train()  # Set back to training mode for next epoch

    print('Finished Training')

    # Plotting training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(epoch_losses)+1), epoch_losses, label='Training Loss', marker='o', linestyle='-', color='b')
    plt.plot(range(1, len(val_epoch_losses)+1), val_epoch_losses, label='Validation Loss', marker='o', linestyle='-', color='r')
    plt.title('Training and Validation Loss Per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # return model



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    file_paths = [
    r"filepath",
    r"filepath",
    r"filepath"   
    ]
    # Load and preprocess the data, then create datasets and data loaders
    X_train, X_test, y_train, y_test = load_and_preprocess_multiple_files(file_paths)


    train_dataset, test_dataset = create_datasets(X_train, X_test, y_train, y_test)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Initialize models, criterion, and optimizer
    mae_model = MAEModel().to(device)
    freeze_parameters(mae_model)  # Freeze the MAE encoder during ViT training
    vit_model = ViTForHeartRateEstimation(mae_model).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = Adam(vit_model.parameters(), lr=5e-5)


    # Training loop
    train_vit(vit_model, train_loader, test_loader, criterion, optimizer, device, epochs=2000, patience=50)


    # Evaluation and plotting
    vit_model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                inputs, labels = batch
            else:
                inputs = batch[0]
                labels = None  # No labels available


            inputs = inputs.to(device)
            outputs = vit_model(inputs).squeeze(1)
            all_predictions.extend(outputs.cpu().numpy())


            if labels is not None:
                labels = labels.to(device)
                all_labels.extend(labels.cpu().numpy())


    # Convert predictions list to a DataFrame
    # predictions_df = pd.DataFrame({'Predicted Heart Rates': all_predictions})
    # if all_labels:
    #     predictions_df['Actual Heart Rates'] = all_labels


if __name__ == '__main__':
    main()




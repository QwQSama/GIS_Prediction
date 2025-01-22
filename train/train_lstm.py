import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from collections import deque
# Get the absolute path of the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from model.lstm_model import LSTM, load_dataset

# Create result directories for each split
model_name = "LSTM"
split_methods = ["random_split", "uniform_split", "time_split"]

# Training function
def train_model(folder_name, result_folder):
    os.makedirs(result_folder, exist_ok=True)

    # Load dataset
    seq_length = 10  # Sequence length for LSTM
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(folder_name, seq_length)

    # Define model, loss function, and optimizer
    input_dim = X_train.shape[2]
    model = LSTM(input_dim=input_dim, hidden_dim=128, num_layers=2, output_dim=1, dropout=0.2)
    criterion = nn.MSELoss()

    # Learning rate schedule
    learning_rates = [0.001, 0.0001, 0.00001]
    stages = [500, 300, 200]
    patience = 10  # Early stopping patience
    improvement_threshold = 1e-4  # Minimum improvement threshold
    ma_window = 5  # Moving average window size

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rates[0])
    current_stage = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    val_losses_ma = deque(maxlen=ma_window)

    train_losses = []
    val_losses = []

    for epoch in range(sum(stages)):
        # Check if we need to switch learning rate stage
        if epoch == sum(stages[:current_stage + 1]):
            current_stage += 1
            if current_stage < len(learning_rates):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rates[current_stage]
                print(f"Switching to learning rate: {learning_rates[current_stage]:.6f}")

        # Training
        model.train()
        permutation = torch.randperm(X_train.size(0))
        epoch_loss = 0
        for i in range(0, X_train.size(0), 64):  # Batch size = 64
            indices = permutation[i:i + 64]
            batch_X, batch_y = X_train[indices], y_train[indices]
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / (X_train.size(0) // 64))

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_losses.append(val_loss)

        # Update moving average of validation losses
        val_losses_ma.append(val_loss)
        ma_val_loss = np.mean(val_losses_ma)

        # Print progress
        print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Moving Avg Val Loss: {ma_val_loss:.4f}")

        # Early stopping condition
        if ma_val_loss > best_val_loss - improvement_threshold:
            epochs_no_improve += 1
        else:
            best_val_loss = ma_val_loss
            epochs_no_improve = 0

        if epochs_no_improve >= patience:
            print(f"No improvement for {patience} epochs. Jumping to next learning rate.")
            current_stage += 1
            if current_stage < len(learning_rates):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = learning_rates[current_stage]
                print(f"Switching to learning rate: {learning_rates[current_stage]:.6f}")
                epochs_no_improve = 0
            else:
                print("No more learning rate stages. Ending training early.")
                break

    # Save training loss plot
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Training and Validation Loss - {model_name}")
    plt.savefig(os.path.join(result_folder, "loss_plot.png"))
    plt.close()

    # Save model and training loss
    torch.save(model.state_dict(), os.path.join(result_folder, f"{model_name.lower()}_model.pth"))
    np.save(os.path.join(result_folder, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(result_folder, "val_losses.npy"), np.array(val_losses))

    print(f"Training completed for {folder_name}. Results saved in {result_folder}")

# Train models for each split method
for split_method in split_methods:
    folder_name = f"../data/{split_method}"
    result_folder = f"{model_name.lower()}_results_{split_method}"
    train_model(folder_name, result_folder)

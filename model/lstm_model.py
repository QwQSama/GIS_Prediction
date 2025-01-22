import torch
import torch.nn as nn
import pandas as pd
import os
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=1, dropout=0.2):
        """
        LSTM Model for time series prediction.
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            output_dim: Number of output features
            dropout: Dropout rate
        """
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Forward pass for LSTM.
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Only take the last time step
        return out

def create_sequences(X, y, seq_length):
    """
    Convert dataset into LSTM-friendly sequences.
    Args:
        X: Input features array
        y: Target values array
        seq_length: Sequence length
    Returns:
        Tuple of tensors (sequences, labels)
    """
    sequences, labels = [], []
    for i in range(len(X) - seq_length):
        sequences.append(X[i:i + seq_length])
        labels.append(y[i + seq_length])
    # Use numpy arrays to improve performance
    sequences = np.array(sequences)
    labels = np.array(labels)
    return torch.tensor(sequences, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32).view(-1, 1)

def load_dataset(folder, seq_length=10):
    """
    Load dataset and transform into LSTM format with sequences.
    Args:
        folder: Folder containing dataset files
        seq_length: Sequence length for LSTM input
    Returns:
        Tuple of tensors (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    X_train = pd.read_csv(os.path.join(folder, "X_train.csv")).values
    y_train = pd.read_csv(os.path.join(folder, "y_train.csv")).values
    X_val = pd.read_csv(os.path.join(folder, "X_val.csv")).values
    y_val = pd.read_csv(os.path.join(folder, "y_val.csv")).values
    X_test = pd.read_csv(os.path.join(folder, "X_test.csv")).values
    y_test = pd.read_csv(os.path.join(folder, "y_test.csv")).values

    X_train, y_train = create_sequences(X_train, y_train, seq_length)
    X_val, y_val = create_sequences(X_val, y_val, seq_length)
    X_test, y_test = create_sequences(X_test, y_test, seq_length)

    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    folder_name = "../data/random_split"  # Choose: "random_split", "uniform_split", or "time_split"
    
    # Load dataset
    seq_length = 10  # LSTM requires sequences of fixed length
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(folder_name, seq_length)

    # Create LSTM model
    input_dim = X_train.shape[2]  # LSTM expects 3D input: [batch_size, seq_length, input_dim]
    model = LSTM(input_dim=input_dim)

    # Print model architecture
    print(model)

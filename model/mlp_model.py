import torch
import torch.nn as nn
import pandas as pd
import os

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=50, output_dim=1):
        super(MLP, self).__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def load_dataset(folder):
    """ Load dataset from the specified folder. """
    X_train = pd.read_csv(os.path.join(folder, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(folder, "y_train.csv"))
    X_val = pd.read_csv(os.path.join(folder, "X_val.csv"))
    y_val = pd.read_csv(os.path.join(folder, "y_val.csv"))
    X_test = pd.read_csv(os.path.join(folder, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(folder, "y_test.csv"))
    
    return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    folder_name = "../data/random_split"  # Choose: "random_split", "uniform_split", or "time_split"
    
    # Load dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(folder_name)

    # Create model
    input_dim = X_train.shape[1]
    model = MLP(input_dim=input_dim, hidden_dim=128, num_layers=50)

    # Print model architecture
    print(model)

import torch
import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Get the absolute path of the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from model.mlp_model import MLP, load_dataset

# Define model name and split methods
model_name = "MLP"
split_methods = ["random_split", "uniform_split", "time_split"]

# Testing function
def test_model(folder_name, model_path, result_folder):
    os.makedirs(result_folder, exist_ok=True)

    # Load dataset
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(folder_name)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Load the trained model
    input_dim = X_test.shape[1]
    model = MLP(input_dim=input_dim, hidden_dim=128, num_layers=50)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Perform prediction
    with torch.no_grad():
        y_pred = model(X_test).numpy()

    # Compute evaluation metrics
    y_test_np = y_test.numpy()
    mse = mean_squared_error(y_test_np, y_pred)
    mae = mean_absolute_error(y_test_np, y_pred)
    rmse = np.sqrt(mse)

    # Save metrics
    metrics = {
        "Mean Squared Error (MSE)": mse,
        "Mean Absolute Error (MAE)": mae,
        "Root Mean Squared Error (RMSE)": rmse,
    }
    metrics_path = os.path.join(result_folder, "metrics.txt")
    with open(metrics_path, "w") as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")

    # Save predictions
    predictions = pd.DataFrame({
        "True Values": y_test_np.flatten(),
        "Predicted Values": y_pred.flatten()
    })
    predictions.to_csv(os.path.join(result_folder, "predictions.csv"), index=False)

    # Plot true vs predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_np, label="True Values", color="blue", linestyle="--")
    plt.plot(y_pred, label="Predicted Values", color="red")
    plt.xlabel("Sample Index")
    plt.ylabel("Values")
    plt.title(f"True vs Predicted Values - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, "true_vs_predicted.png"))
    plt.close()

    print(f"Testing completed for {folder_name}. Results saved in {result_folder}")

# Test models for each split method
for split_method in split_methods:
    folder_name = f"../data/{split_method}"
    model_path = f"../train/{model_name.lower()}_results_{split_method}/{model_name.lower()}_model.pth"
    result_folder = f"{model_name.lower()}_results_{split_method}"
    test_model(folder_name, model_path, result_folder)

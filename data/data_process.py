import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import joblib  

# Load dataset
file_path = "gisdata_qwq.xlsx"
df = pd.read_excel(file_path)

# Remove empty columns
df = df.drop(columns=["Unnamed: 3"], errors="ignore")

# Convert object columns to float
for col in ["huminity", "cloudBaseHeight"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove rows with missing values
df = df.dropna().reset_index(drop=True)

# Detect and remove outliers using Z-score
def remove_outliers(df, columns, threshold=3):
    """
    Remove rows with outliers based on Z-score.
    Args:
        df: DataFrame
        columns: List of columns to check for outliers
        threshold: Z-score threshold (default=3)
    Returns:
        Cleaned DataFrame
    """
    z_scores = df[columns].apply(zscore)
    mask = (z_scores.abs() < threshold).all(axis=1)
    return df[mask]

# Apply outlier removal
columns_to_check = [
    "GIS3_2", "GIS4_2", "temp2m", "rain", "huminity", 
    "cloudBaseHeight", "totalCloudCover", "radiation", 
    "cloudlow", "cloudmiddle", "cloudhigh"
]
df = remove_outliers(df, columns=columns_to_check)

# Display dataset information
print("Cleaned Dataset Information:")
print(df.info())

# Define features and target
features = [
    "temp2m", "rain", "huminity", "cloudBaseHeight",
    "totalCloudCover", "radiation", "cloudlow", "cloudmiddle", "cloudhigh"
]
target = "GIS4_2"

X = df[features]
y = df[target]

# Compute dataset sizes
total_size = len(df)
train_size = int(total_size * 0.7)
val_size = int(total_size * 0.15)
test_size = total_size - train_size - val_size

# Function to save datasets and scalers in a specific folder
def save_dataset_and_scaler(folder, X_train, y_train, X_val, y_val, X_test, y_test):
    os.makedirs(folder, exist_ok=True)

    # Save scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(folder, "scaler.pkl"))

    # Save standardized datasets
    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(os.path.join(folder, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(folder, "y_train.csv"), index=False)
    pd.DataFrame(X_val_scaled, columns=X_val.columns).to_csv(os.path.join(folder, "X_val.csv"), index=False)
    y_val.to_csv(os.path.join(folder, "y_val.csv"), index=False)
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(os.path.join(folder, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(folder, "y_test.csv"), index=False)

    print(f"Data and scaler saved in: {folder}")

# Random split
X_train_rnd, X_temp_rnd, y_train_rnd, y_temp_rnd = train_test_split(X, y, train_size=train_size, random_state=42)
X_val_rnd, X_test_rnd, y_val_rnd, y_test_rnd = train_test_split(X_temp_rnd, y_temp_rnd, test_size=test_size, random_state=42)
save_dataset_and_scaler("random_split", X_train_rnd, y_train_rnd, X_val_rnd, y_val_rnd, X_test_rnd, y_test_rnd)

# Uniform interval split
indices = np.arange(total_size)
train_indices = indices[np.linspace(0, total_size - 1, train_size, dtype=int)]
val_indices = indices[np.linspace(1, total_size - 1, val_size, dtype=int)]
test_indices = indices[np.linspace(2, total_size - 1, test_size, dtype=int)]

X_train_seq = X.iloc[train_indices]
y_train_seq = y.iloc[train_indices]
X_val_seq = X.iloc[val_indices]
y_val_seq = y.iloc[val_indices]
X_test_seq = X.iloc[test_indices]
y_test_seq = y.iloc[test_indices]
save_dataset_and_scaler("uniform_split", X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq)

# Sliding window split
X_train_time = X.iloc[:train_size]
y_train_time = y.iloc[:train_size]
X_val_time = X.iloc[train_size:train_size + val_size]
y_val_time = y.iloc[train_size:train_size + val_size]
X_test_time = X.iloc[train_size + val_size:]
y_test_time = y.iloc[train_size + val_size:]
save_dataset_and_scaler("time_split", X_train_time, y_train_time, X_val_time, y_val_time, X_test_time, y_test_time)

# Print dataset sizes
print("\nDataset splitting completed.")
print("Random split: Train =", len(X_train_rnd), "Validation =", len(X_val_rnd), "Test =", len(X_test_rnd))
print("Uniform interval split: Train =", len(X_train_seq), "Validation =", len(X_val_seq), "Test =", len(X_test_seq))
print("Sliding window split: Train =", len(X_train_time), "Validation =", len(X_val_time), "Test =", len(X_test_time))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, message=".*is_sparse is deprecated.*")

# Load datasets
Dataset_path = "/home/pegah/Codes/imd/"
Dataset_names = ["Op1_CapDL_NoSteps.csv", "Op1_CapDL_NoSteps_wo_correlated_features.csv", "outliers_and_anomalies.csv"]

# Load ground truth labels
df2 = pd.read_csv(Dataset_path + Dataset_names[2], header=0, sep=',')

# Load main dataset with features
df = pd.read_csv(Dataset_path + Dataset_names[0], header=0, sep=',')

# Combine datasets
df_new = pd.concat([df, df2.iloc[:, 1:]], axis=1)

# Display dataset information
#print(df_new.shape)
#print(df_new.columns)
print(f"Number of outliers: {df_new[df_new['outliers'] == 1.0].shape[0]}")

# Separate normal and anomalous data
normal_data = df_new[df_new['outliers'] == 0.0].drop(['Unnamed: 0', 'outliers'], axis=1)
anomaly_data = df_new[df_new['outliers'] == 1.0].drop(['Unnamed: 0', 'outliers'], axis=1)

# Normalize data
scaler = MinMaxScaler()
normal_data_scaled = scaler.fit_transform(normal_data)

# Split into train and validation sets
X_train, X_val = train_test_split(normal_data_scaled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

# Define the AutoEncoder model
class AutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Initialize the model
input_dim = X_train.shape[1]
model = AutoEncoder(input_dim)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, X_train_tensor)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

# Prepare test data (both normal and anomalous)
test_data = df_new.drop(['Unnamed: 0', 'outliers'], axis=1)
test_labels = df_new['outliers']
test_data_scaled = scaler.transform(test_data)
test_tensors = torch.tensor(test_data_scaled, dtype=torch.float32)

# Calculate reconstruction errors
model.eval()
with torch.no_grad():
    reconstructed = model(test_tensors)
    reconstruction_errors = torch.mean((reconstructed - test_tensors) ** 2, dim=1).numpy()

# Add reconstruction errors to the DataFrame
df_new['reconstruction_error'] = reconstruction_errors

# Determine threshold based on validation errors
with torch.no_grad():
    val_reconstructed = model(X_val_tensor)
    val_errors = torch.mean((val_reconstructed - X_val_tensor) ** 2, dim=1).numpy()

# Fix potential numerical issues in val_errors
val_errors = np.nan_to_num(val_errors, nan=0.0, posinf=np.inf, neginf=-np.inf)
threshold = np.percentile(val_errors, 95)  # 95th percentile threshold
print(f"Fixed Threshold for anomaly detection: {threshold:.4f}")

# Classify anomalies based on the threshold
df_new['predicted_label'] = (df_new['reconstruction_error'] > threshold).astype(int)

# Evaluate the model
"""
print("\nClassification Report:")
print(classification_report(test_labels, df_new['predicted_label']))

print("Confusion Matrix:")
print(confusion_matrix(test_labels, df_new['predicted_label']))
"""
print("\n AUROC Evaluation:")
print(roc_auc_score(test_labels.values, df_new['predicted_label'].values))

# K-means for Threshold
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(reconstruction_errors.reshape(-1, 1))
# Find the cluster with the higher mean error (the anomaly cluster)
anomaly_cluster_kmeans = np.argmax(kmeans.cluster_centers_)  # Cluster with higher mean error
kmeans_threshold = np.percentile(reconstruction_errors[kmeans_labels != anomaly_cluster_kmeans], 95)
print(f"K-means Threshold: {kmeans_threshold:.4f}")

# Apply K-means Threshold
df_new['predicted_label_kmeans'] = (df_new['reconstruction_error'] > kmeans_threshold).astype(int)
#print("\nClassification Report (K-means Threshold):")
#print(classification_report(test_labels, df_new['predicted_label_kmeans']))
print("\n AUROC Evaluation:")
print(roc_auc_score(test_labels.values, df_new['predicted_label_kmeans'].values))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans, DBSCAN

# Load Dataset
Dataset_path = "/home/pegah/Codes/imd/"
Dataset_names = ["Op1_CapDL_NoSteps.csv", "Op1_CapDL_NoSteps_wo_correlated_features.csv", "outliers_and_anomalies.csv"]

df2 = pd.read_csv(Dataset_path + Dataset_names[2], header=0, sep=',')
df = pd.read_csv(Dataset_path + Dataset_names[0], header=0, sep=',')

df_new = pd.concat([df, df2.iloc[:, 1:]], axis=1)
normal_data = df_new[df_new['outliers'] == 0.0].drop(['Unnamed: 0', 'outliers'], axis=1)

X_train, X_val = train_test_split(normal_data, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)

# AutoEncoder Class
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

input_dim = X_train.shape[1]
model = AutoEncoder(input_dim)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, X_train_tensor)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item()}")

# Combine normal and anomalous data
test_data = df_new.drop(['Unnamed: 0', 'outliers'], axis=1)
test_labels = df_new['outliers']
test_tensors = torch.tensor(test_data.values, dtype=torch.float32)

# Calculate reconstruction error
model.eval()
with torch.no_grad():
    reconstructed = model(test_tensors)
    reconstruction_error = torch.mean((reconstructed - test_tensors) ** 2, dim=1).numpy()
    df_new['reconstruction_error'] = reconstruction_error

    # Fixed Threshold
    val_reconstructed = model(X_val_tensor)
    val_errors = torch.mean((val_reconstructed - X_val_tensor) ** 2, dim=1).numpy()
    fixed_threshold = val_errors.mean() + 2 * val_errors.std()
    print(f"Fixed Threshold: {fixed_threshold:.4f}")

    # K-means for Threshold
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_labels = kmeans.fit_predict(reconstruction_error.reshape(-1, 1))
    # Find the cluster with the higher mean error (the anomaly cluster)
    anomaly_cluster_kmeans = np.argmax(kmeans.cluster_centers_)  # Cluster with higher mean error
    kmeans_threshold = np.percentile(reconstruction_error[kmeans_labels != anomaly_cluster_kmeans], 95)
    print(f"K-means Threshold: {kmeans_threshold:.4f}")

    # DBSCAN for Threshold
    dbscan = DBSCAN(eps=0.05, min_samples=5)  # Adjust `eps` and `min_samples` as needed
    dbscan_labels = dbscan.fit_predict(reconstruction_error.reshape(-1, 1))
    # Get reconstruction errors for points that are not labeled as outliers (-1)
    normal_data_errors = reconstruction_error[dbscan_labels != -1]  # Ignore outliers marked as -1
    dbscan_threshold = np.percentile(normal_data_errors, 95) if len(normal_data_errors) > 0 else float('inf')
    print(f"DBSCAN Threshold: {dbscan_threshold:.4f}")

# Apply Fixed Threshold
df_new['predicted_label_fixed'] = (df_new['reconstruction_error'] > fixed_threshold).astype(int)

# Apply K-means Threshold
df_new['predicted_label_kmeans'] = (df_new['reconstruction_error'] > kmeans_threshold).astype(int)

# Apply DBSCAN Threshold
df_new['predicted_label_dbscan'] = (df_new['reconstruction_error'] > dbscan_threshold).astype(int)

# Classification Reports and Confusion Matrices
print("\nClassification Report (Fixed Threshold):")
print(classification_report(test_labels, df_new['predicted_label_fixed']))

print("\nClassification Report (K-means Threshold):")
print(classification_report(test_labels, df_new['predicted_label_kmeans']))

print("\nClassification Report (DBSCAN Threshold):")
print(classification_report(test_labels, df_new['predicted_label_dbscan']))

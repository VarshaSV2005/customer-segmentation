# Ensure that we use the 'Agg' backend for non-GUI plotting
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for saving plots without GUI

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Generate Synthetic Dataset
np.random.seed(42)

data = {
    "CustomerID": range(1, 16),  # 15 customers
    "TotalSpend": np.random.randint(100, 1000, 15),
    "PurchaseFrequency": np.random.randint(1, 50, 15),
    "AverageOrderValue": np.random.uniform(20, 200, 15).round(2),
}

df = pd.DataFrame(data)

# Introduce some missing values
df.loc[np.random.choice(df.index, 3, replace=False), "TotalSpend"] = np.nan
df.loc[np.random.choice(df.index, 2, replace=False), "PurchaseFrequency"] = np.nan

# Save the dataset as a CSV file
df.to_csv("customer_data.csv", index=False)

# Step 2: Preprocessing
# Load dataset (if starting from the CSV)
df = pd.read_csv("customer_data.csv")

# Handle missing values: Replace with the column mean
df.fillna(df.mean(), inplace=True)

# Drop non-numeric columns if present
numeric_df = df.drop("CustomerID", axis=1)

# Standardize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_df)

# Step 3: Dimensionality Reduction with PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Convert PCA results to DataFrame for plotting
pca_df = pd.DataFrame(pca_data, columns=["PC1", "PC2"])

# Step 4: K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_data)

# Add cluster labels to PCA DataFrame
pca_df["KMeans_Cluster"] = kmeans_labels

# Step 5: Agglomerative Clustering (Fixed)
agglo = AgglomerativeClustering(n_clusters=3, linkage="ward")
agglo_labels = agglo.fit_predict(scaled_data)

# Add cluster labels to PCA DataFrame
pca_df["Agglo_Cluster"] = agglo_labels

# Step 6: Visualization
# Scatter Plot for K-Means Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x="PC1", y="PC2", hue="KMeans_Cluster", data=pca_df, palette="viridis", s=100)
plt.title("K-Means Clustering (PCA Projection)", fontsize=14)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Cluster")
plt.savefig("kmeans_clusters.png")  # Save visualization
plt.close()  # Close the plot to free up memory

# Dendrogram for Agglomerative Clustering
plt.figure(figsize=(10, 6))
linkage_matrix = linkage(scaled_data, method="ward")
dendrogram(linkage_matrix)
plt.title("Dendrogram (Agglomerative Clustering)", fontsize=14)
plt.xlabel("Customer Index")
plt.ylabel("Distance")
plt.savefig("dendrogram.png")  # Save dendrogram
plt.close()  # Close the plot to free up memory

# Save clustering results
pca_df.to_csv("customer_segments.csv", index=False)
print("Clustering complete. Results saved as 'customer_segments.csv'.")

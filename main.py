# main.py

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Step 1: Paths
# -------------------------------
DATA_FOLDER = "data"
OUTPUT_FOLDER = "outputs"
PLOTS_FOLDER = os.path.join(OUTPUT_FOLDER, "plots")
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

INPUT_CSV = os.path.join(DATA_FOLDER, "train.csv")
OUTPUT_CSV = os.path.join(DATA_FOLDER, "customer_segments_final.csv")
PLOT_PATH = os.path.join(PLOTS_FOLDER, "final_clusters.png")

# -------------------------------
# Step 2: Load dataset
# -------------------------------
df = pd.read_csv(INPUT_CSV)
print("Dataset loaded. First 5 rows:")
print(df.head())

# -------------------------------
# Step 3: Preprocess
# -------------------------------
# Convert Gender to numeric
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Select features for clustering
X = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nFeatures scaled.")

# -------------------------------
# Step 4: Train KMeans
# -------------------------------
# Choose number of clusters (from Elbow method)
K = 5
kmeans = KMeans(n_clusters=K, random_state=42)
kmeans.fit(X_scaled)

# Assign clusters to dataframe
df['Cluster'] = kmeans.labels_

# Map cluster numbers to human-readable names
cluster_names = {
    0: 'Young Big Spenders',
    1: 'Young Medium Spenders',
    2: 'Middle Age Average',
    3: 'Older Low Spenders',
    4: 'Middle Age Big Spenders'
}
df['Cluster Name'] = df['Cluster'].map(cluster_names)
print("\nCluster counts:")
print(df['Cluster Name'].value_counts())

# -------------------------------
# Step 5: Save final dataset
# -------------------------------
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nFinal dataset saved to: {OUTPUT_CSV}")

# -------------------------------
# Step 6: Visualize clusters
# -------------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    hue='Cluster Name',
    palette='Set2'
)
plt.title("Customer Segments")
plt.savefig(PLOT_PATH)
plt.close()  # closes figure so script continues
print(f"Cluster plot saved to: {PLOT_PATH}")

print("\nAll steps completed successfully!")
from sklearn.cluster import KMeans
import pandas as pd

def train_kmeans(df_scaled, n_clusters=5):
    """Train K-Means and return model"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(df_scaled)
    return kmeans

def assign_clusters(df, kmeans):
    """Assign cluster labels to original dataframe"""
    df['Cluster'] = kmeans.labels_
    return df

def add_cluster_names(df):
    """Map cluster numbers to friendly names"""
    cluster_name_map = {
        0: 'Young Big Spenders',
        1: 'Young Big Spenders',
        2: 'Young Medium Spenders',
        3: 'Young Medium Spenders',
        4: 'Young Medium Spenders'
    }
    df['Cluster Name'] = df['Cluster'].map(cluster_name_map)
    return df
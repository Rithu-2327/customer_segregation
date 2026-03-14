import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load CSV dataset"""
    df = pd.read_csv(file_path)
    return df

def scale_features(df, feature_cols):
    """Scale numerical features"""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])
    df_scaled = pd.DataFrame(scaled_features, columns=feature_cols)
    return df_scaled
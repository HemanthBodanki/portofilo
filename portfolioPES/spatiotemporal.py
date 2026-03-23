import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import folium
from datetime import datetime

# -----------------------------
# STEP 1: LOAD DATA
# -----------------------------
# CSV format required:
# latitude, longitude, date
# Example:
# 17.3850, 78.4867, 2024-01-15

def load_data(file_path):
    df = pd.read_csv(file_path)

    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    return df


# -----------------------------
# STEP 2: FEATURE ENGINEERING
# -----------------------------
def prepare_features(df):
    # Convert date to numeric (timestamp)
    df['timestamp'] = df['date'].astype('int64') // 10**9

    # Combine spatial + temporal features
    features = df[['latitude', 'longitude', 'timestamp']]

    return features


# -----------------------------
# STEP 3: NORMALIZATION
# -----------------------------
def normalize_features(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    return scaled_features


# -----------------------------
# STEP 4: APPLY DBSCAN
# -----------------------------
def apply_clustering(features, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(features)

    return labels


# -----------------------------
# STEP 5: ADD CLUSTERS TO DATA
# -----------------------------
def assign_clusters(df, labels):
    df['cluster'] = labels
    return df


# -----------------------------
# STEP 6: ANALYSIS
# -----------------------------
def analyze_clusters(df):
    print("\n📊 Cluster Summary:\n")
    cluster_counts = df['cluster'].value_counts()

    print(cluster_counts)

    print("\nHigh Risk Zones (excluding noise -1):")
    high_risk = df[df['cluster'] != -1]['cluster'].value_counts()

    print(high_risk)


# -----------------------------
# STEP 7: VISUALIZATION (MAP)
# -----------------------------
def create_map(df, output_file="collision_map.html"):
    # Center map
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

    # Color palette
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred']

    for _, row in df.iterrows():
        cluster = row['cluster']
        color = 'black' if cluster == -1 else colors[cluster % len(colors)]

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"Cluster: {cluster}\nDate: {row['date']}"
        ).add_to(m)

    m.save(output_file)
    print(f"\n🗺️ Map saved as {output_file}")


# -----------------------------
# STEP 8: MAIN FUNCTION
# -----------------------------
def main():
    file_path = "wildlife_collisions.csv"

    print("📥 Loading data...")
    df = load_data(file_path)

    print("⚙️ Preparing features...")
    features = prepare_features(df)

    print("📊 Normalizing data...")
    scaled_features = normalize_features(features)

    print("🤖 Running DBSCAN clustering...")
    labels = apply_clustering(scaled_features, eps=0.7, min_samples=4)

    df = assign_clusters(df, labels)

    analyze_clusters(df)

    create_map(df)


# -----------------------------
# RUN PROGRAM
# -----------------------------
if __name__ == "__main__":
    main()
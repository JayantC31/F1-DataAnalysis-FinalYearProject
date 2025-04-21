import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from streamlit_app import getRaceData
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# get num of turns per circuit
def count_turns(lap_df, angle_threshold=10):
    """
    Counts the number of turns in a lap based on changes in the coordinates by a certain amounnt of degrees change

    """
    lap_df = lap_df.sort_index()
    dx = lap_df["X"].diff()
    dy = lap_df["Y"].diff()
    headings = np.degrees(np.arctan2(dy, dx))
    d_heading = headings.diff().abs()
    d_heading = d_heading.apply(lambda x: x if x < 180 else 360 - x)
    turns = 0
    in_turn = False
    for dh in d_heading.fillna(0):
        if not in_turn and dh > angle_threshold:
            turns += 1
            in_turn = True
        elif in_turn and dh < angle_threshold:
            in_turn = False
    return turns

# get data specific on straights per circuit
def get_straight_metrics(lap_df, angle_threshold=5):
    """
    
    a straight segment is defined where the change in heading of car is below the threshold of a turn
   
    """
    lap_df = lap_df.sort_index()
    dx = lap_df["X"].diff()
    dy = lap_df["Y"].diff()
    distances = np.sqrt(dx**2 + dy**2)
    
    headings = np.degrees(np.arctan2(dy, dx))
    d_heading = headings.diff().abs().fillna(0)
    d_heading = d_heading.apply(lambda x: x if x < 180 else 360 - x)
    
    max_straight = 0
    total_straight = 0
    current_straight = 0
    for i in range(1, len(lap_df)):
        if d_heading.iloc[i] < angle_threshold:
            current_straight += distances.iloc[i]
        else:
            total_straight += current_straight
            if current_straight > max_straight:
                max_straight = current_straight
            current_straight = 0
    total_straight += current_straight
    if current_straight > max_straight:
        max_straight = current_straight
    return max_straight, total_straight


def compute_gear_distribution(lap_df, gear_range=range(1, 9)):
    """
    Computes the gear distribution per part of lap as the ratio of data points in each gear
    """
    gear_counts = lap_df["nGear_x"].value_counts(normalize=True)
    distribution = {}
    for gear in gear_range:
        distribution[f"gear{gear}_ratio"] = gear_counts.get(gear, 0)
    return distribution

def aggregate_weather_features(weather_df):
    """
    get weather data for each race

    """
    weather_columns = ['Humidity', 'Pressure']
    weather_agg = weather_df.groupby("race")[weather_columns].mean().reset_index()
    return weather_agg



def preprocessTelemetryData():
    """
    preprocess telemetry data per car per circuit
        """
    lap_summary_df, telemetry_df, weather_df = getRaceData(2024)

    # sort by race
    lap_summary_df = lap_summary_df.sort_values(by='race')
    telemetry_df = telemetry_df.sort_values(by='race')
    
    # use time as primary value
    time_columns = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]
    for col in time_columns:
        try:
            lap_summary_df[col] = pd.to_timedelta(lap_summary_df[col]).dt.total_seconds()
        except Exception as e:
            print(f"Error converting column {col} to seconds: {e}")
    
   # remove outliers on some columns
    outlier_columns = ["LapTime", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]
    for col in outlier_columns:
        try:
            filtered_groups = []
            for race, group in lap_summary_df.groupby("race"):
                lower = group[col].quantile(0.05)
                upper = group[col].quantile(0.95)
                filtered_group = group[(group[col] > lower) & (group[col] < upper)]
                filtered_groups.append(filtered_group)
            lap_summary_df = pd.concat(filtered_groups, ignore_index=True)
        except Exception as e:
            print(f"Error removing outliers from column {col}: {e}")
    
    # get mean values for some columns
    agg_columns = ["LapTime", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST", "TyreLife"]
    circuit_agg = lap_summary_df.groupby("race")[agg_columns].mean().reset_index()
    
    # scale and clean the values
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(circuit_agg[agg_columns])
    circuit_features_scaled_df = pd.DataFrame(scaled_features, columns=agg_columns)
    circuit_features_scaled_df.insert(0, "race", circuit_agg["race"])
    
    return circuit_features_scaled_df, lap_summary_df, telemetry_df, weather_df

def aggregate_telemetry_features(telemetry_df):
    """
    make new features and then aggregate some by getting means etc
    
    
    """
    aggregated = []
    
    for race, race_group in telemetry_df.groupby("race"):
        race_dict = {"race": race}
        brake_events_list = []          # Brake events count 
        throttle_means = []             # Mean throttle usage
        track_length_list = []          # Track length 
        max_straight_list = []          # Maximum straight length per lap
        total_straight_list = []        # Total straight distance per lap
        elevation_means = []            # Average elevation per lap
        elevation_variability_list = [] # elevation variable 
        turns_list = []                 #  num of turns 
        rpm_mean_list = []              # avg RPM 
        rpm_std_list = []               # RPM variability 
        gear_change_frequency_list = [] # gear change 
        # gears between 1 and 8 
        gear_distribution_agg = {f"gear{g}_ratio": [] for g in range(1, 9)}
        
        for lap, lap_df in race_group.groupby("lap"):
            
            brake_events = lap_df["Brake_x"].astype(int).sum()
            brake_events_list.append(brake_events)
            
            throttle_means.append(lap_df["Throttle_x"].mean())
            
            track_length_list.append(lap_df["Distance"].max())
            
            max_straight, total_straight = get_straight_metrics(lap_df, angle_threshold=5)
            max_straight_list.append(max_straight)
            total_straight_list.append(total_straight)
            
            elevation_means.append(lap_df["Z"].mean())
            elevation_variability_list.append(lap_df["Z"].std())
            
            turns_list.append(count_turns(lap_df, angle_threshold=10))
            
            if "RPM_x" in lap_df.columns:
                rpm_mean_list.append(lap_df["RPM_x"].mean())
                rpm_std_list.append(lap_df["RPM_x"].std())
            else:
                rpm_mean_list.append(np.nan)
                rpm_std_list.append(np.nan)
            
            gear_changes = lap_df["nGear_x"].diff().fillna(0)
            gear_change_frequency_list.append((gear_changes != 0).sum())
            
            gear_dist = compute_gear_distribution(lap_df, gear_range=range(1, 9))
            for key, value in gear_dist.items():
                gear_distribution_agg[key].append(value)
        
        race_dict["avg_num_of_brake_events"] = np.mean(brake_events_list) if brake_events_list else np.nan
        race_dict["avg_throttle_usage"] = np.mean(throttle_means) if throttle_means else np.nan
        race_dict["track_length"] = np.mean(track_length_list) if track_length_list else np.nan
        race_dict["max_straight_length"] = np.max(max_straight_list) if max_straight_list else np.nan
        race_dict["total_straight_distance"] = np.mean(total_straight_list) if total_straight_list else np.nan
        race_dict["avg_elevation"] = np.mean(elevation_means) if elevation_means else np.nan
        race_dict["elevation_variability"] = np.mean(elevation_variability_list) if elevation_variability_list else np.nan
        race_dict["avg_turns"] = np.mean(turns_list) if turns_list else np.nan
        race_dict["avg_rpm"] = np.mean(rpm_mean_list) if rpm_mean_list else np.nan
        race_dict["rpm_variability"] = np.mean(rpm_std_list) if rpm_std_list else np.nan
        race_dict["avg_gear_change_frequency"] = np.mean(gear_change_frequency_list) if gear_change_frequency_list else np.nan
        
        for key, values in gear_distribution_agg.items():
            race_dict[key] = np.mean(values) if values else np.nan
        
        aggregated.append(race_dict)
    
    return pd.DataFrame(aggregated)

def processRaceData():
    """
    Preprocesses race data and merge with weather
    
   
    """
    circuit_features_scaled_df, cleaned_lap_summary, cleaned_telemetry, weather_df = preprocessTelemetryData()
    
    aggregated_telemetry_df = aggregate_telemetry_features(cleaned_telemetry)
    
    # scale with telemetry
    telemetry_feature_columns = aggregated_telemetry_df.columns.drop("race")
    scaler_telemetry = StandardScaler()
    aggregated_telemetry_df[telemetry_feature_columns] = scaler_telemetry.fit_transform(
        aggregated_telemetry_df[telemetry_feature_columns]
    )
    
    # merge with aggregate features
    merged_df = pd.merge(
        circuit_features_scaled_df,
        aggregated_telemetry_df,
        on="race",
        how="left"
    )
    
    
    aggregated_weather_df = aggregate_weather_features(weather_df)
    # scale weather
    aggregated_weather_df["race"] = aggregated_weather_df["race"].str.replace(" ", "_")
    scaler_weather = StandardScaler()
    weather_feature_columns = aggregated_weather_df.columns.drop("race")
    aggregated_weather_df[weather_feature_columns] = scaler_weather.fit_transform(
        aggregated_weather_df[weather_feature_columns]
    )
    
    
    final_df = pd.merge(
        merged_df,
        aggregated_weather_df,
        on="race",
        how="inner"
    )
    
    print(final_df.head(24))
    print(final_df.columns)
    return final_df

def runPCA():
    """
    reduce to 3 pca components and display
    
    
    """
    circuit_features_scaled_df = processRaceData()
    
    
    features = circuit_features_scaled_df.drop(columns='race')
    
    features_complete = features.dropna()
    if features_complete.empty:
        raise ValueError("No complete rows found after dropping missing values.")
    
    circuit_features_scaled_df = circuit_features_scaled_df.loc[features_complete.index]
    features = features_complete.copy()
    # 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(features)
    
    circuit_features_scaled_df['PC1'] = pca_result[:, 0]
    circuit_features_scaled_df['PC2'] = pca_result[:, 1]
    circuit_features_scaled_df['PC3'] = pca_result[:, 2]
    
    # print the explained vbariance ratio
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)
    variance_ratio = pca.explained_variance_ratio_
    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=features.columns)
    # test
    print("PCA Loadings:")
    print(loadings)
    print(circuit_features_scaled_df.head())
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(circuit_features_scaled_df['PC1'],
               circuit_features_scaled_df['PC2'],
               circuit_features_scaled_df['PC3'],
               c='blue', alpha=0.7)
    
   # annotate per race
    for i, race in enumerate(circuit_features_scaled_df['race']):
        new_race = race.replace("Grand_Prix", "GP").replace("_", " ")
        offset_x, offset_y = 0.02, 0.02
        ax.text(circuit_features_scaled_df['PC1'].iloc[i] + offset_x,
                circuit_features_scaled_df['PC2'].iloc[i] + offset_y,
                circuit_features_scaled_df['PC3'].iloc[i],
                new_race,
                fontsize=6, color='black')
    
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("3D PCA of Circuit Features")
    
    ax.view_init(elev=15, azim=140)
    ax.dist = 20
    
    return fig, circuit_features_scaled_df, variance_ratio, loadings

def runKMeans():
    """
    use clustering on pca graph
    
    """
    fig_pca, pca_df, _, _ = runPCA()
    n_clusters = 12
    features = pca_df[['PC1', 'PC2', 'PC3']].values
    # use kmeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)
    pca_df['Cluster'] = clusters
    
    cluster_dict = {}
    for cluster in sorted(pca_df['Cluster'].unique()):
        races = pca_df.loc[pca_df['Cluster'] == cluster, 'race'].tolist()
        races = [race.replace("Grand_Prix", "GP").replace("_", " ") for race in races]
        cluster_dict[cluster] = races
    
    print("Clusters and their races:")
    for cluster, races in cluster_dict.items():
        print(f"Cluster {cluster}: {races}")
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        pca_df['PC1'],
        pca_df['PC2'],
        pca_df['PC3'],
        c=pca_df['Cluster'],
        cmap='viridis',
        alpha=0.7
    )
    
    for i, race in enumerate(pca_df['race']):
        new_race = race.replace("Grand_Prix", "GP").replace("_", " ")
        offset_x, offset_y = 0.02, 0.02
        ax.text(
            pca_df['PC1'].iloc[i] + offset_x,
            pca_df['PC2'].iloc[i] + offset_y,
            pca_df['PC3'].iloc[i],
            new_race,
            fontsize=6,
            color='black'
        )
    
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    # set title
    ax.set_title("3D PCA with KMeans Clustering", loc='right')
    
    ax.view_init(elev=15, azim=140)
    ax.dist = 20
    
    centers = kmeans.cluster_centers_
    ax.scatter(
        centers[:, 0], centers[:, 1], centers[:, 2],
        marker='X', s=200, c=(1, 0, 0, 0.3), label='Centroids'
    )
    
    cluster_text = ""
    for cluster in sorted(cluster_dict.keys()):
        cluster_text += f"Cluster {cluster}: " + ", ".join(cluster_dict[cluster]) + "\n"
    fig.text(0.05, 0.95, cluster_text, fontsize=10, va='top', ha='left',
             bbox=dict(facecolor='white', alpha=0.7))
    
    ax.legend()
    
    return fig, pca_df

def runElbowMethod():
    """
    use elbow to see how many clusters to use from pca
    """
    _, pca_df, _, _ = runPCA()
    max_clusters = 24
    features = pca_df[['PC1', 'PC2', 'PC3']].values
    
    getdifference = []
    cluster_range = list(range(1, max_clusters + 1))
    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        getdifference.append(kmeans.inertia_)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cluster_range, getdifference, marker='o')
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("difference (Sum of Squared Distances)")
    ax.set_title("Elbow Method For Optimal Number of Clusters")
    
    return fig

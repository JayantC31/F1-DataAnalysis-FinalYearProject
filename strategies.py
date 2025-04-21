import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

# get the data from csvs
def getDataFromCsv(settings):
    """
    Run the driver analysis based on the settings that are given from the user
    """
    base_folder = "Telemetry_Data"
    year = settings.get("Year")
    race_event = settings.get("Race Event")
    drivers = settings.get("Drivers", [])
    CSVDATA = {}

    for driver in drivers:
        # Construct the full path to the driver's lap summary file
        driver_path = os.path.join(base_folder, year, driver, race_event)
        laps_csv_file = os.path.join(driver_path, "R_laps.csv")
        
        # Check if the file exists before trying to open it
        if os.path.exists(laps_csv_file):
            try:
                df = pd.read_csv(laps_csv_file)
                CSVDATA[driver] = df
            except Exception as e:
                print(f"Error reading {laps_csv_file} for {driver}: {e}")
        else:
            print(f"File not found: {laps_csv_file}")

    return CSVDATA

# create violin plots
def violinplots(csv_data, metrics=["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]):
    """
    get the data from the csv files and plot the violin plots for the given metrics but based on the stint
    """
    # this is needed to check for the metric, sometimes code doesmnt work if metric not a string
    if isinstance(metrics, str):
        metrics = [metrics]
        
    # colour map for the drivers
    driver_names = sorted(csv_data.keys())
    cmap = plt.get_cmap("tab10")
    driver_colors = {driver: cmap(i % 10) for i, driver in enumerate(driver_names)}
    # remove the outliers
    def remove_outliers(numeric_data):
        # get iqr
        Q1 = numeric_data.quantile(0.25)
        Q3 = numeric_data.quantile(0.75)
        IQR = Q3 - Q1
        return numeric_data[(numeric_data >= (Q1 - 1.5 * IQR)) & (numeric_data <= (Q3 + 1.5 * IQR))]
    
    # replace compound names so it looks better
    def format_compound(compound):
        compound = str(compound).upper()
        if compound == "MEDIUM":
            return "M"
        elif compound == "HARD":
            return "H"
        elif compound == "SOFT":
            return "S"
        return compound

    def get_numeric_times(series):
        return pd.to_timedelta(series.astype(str).str.strip(), errors='coerce').dropna().dt.total_seconds()

    # compute the driver stats for the given csv data
    def compute_driver_stats():
        stats = {}
        for driver, df in csv_data.items():
            if "Stint" not in df.columns or "LapTime" not in df.columns:
                continue
            for stint in df["Stint"].unique():
                df_stint = df[df["Stint"] == stint]
                if df_stint.empty:
                    continue
                lap_count = len(df_stint)
                lap_times = get_numeric_times(df_stint["LapTime"])
                if lap_times.empty:
                    continue
                avg_laptime = lap_times.mean()
                comp = df_stint["Compound"].iloc[0] if "Compound" in df_stint.columns else "N/A"
                stats[(driver, stint)] = (lap_count, avg_laptime, format_compound(comp))
        return stats

    def build_group_data(stats):
        group = {}
        all_stints = set()
        for df in csv_data.values():
            if "Stint" in df.columns:
                all_stints.update(df["Stint"].unique())
        for stint in sorted(all_stints):
            group[stint] = {}
            for metric in metrics:
                entries = []
                for driver, df in csv_data.items():
                    if "Stint" not in df.columns or metric not in df.columns:
                        continue
                    df_stint = df[df["Stint"] == stint]
                    if df_stint.empty:
                        continue
                    numeric_data = get_numeric_times(df_stint[metric])
                    if numeric_data.empty:
                        continue
                    # Call the global remove_outliers function.
                    filtered = remove_outliers(numeric_data)
                    if filtered.empty:
                        continue
                    ds = stats.get((driver, stint))
                    if ds is None:
                        continue
                    entries.append((driver, filtered, ds))
                group[stint][metric] = entries
        return group

    
    driver_stint_stats = compute_driver_stats()
    group_data = build_group_data(driver_stint_stats)
    
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 6 * n_metrics), squeeze=False)
    group_gap = 1.0
    
    for m_idx, metric in enumerate(metrics):
        ax = axes[m_idx, 0]
        positions, data_arrays, x_labels, annotations = [], [], [], []
        group_boundaries = []
        current_pos = 1
        for stint in sorted(group_data.keys()):
            entries = group_data[stint].get(metric, [])
            if len(entries) < 2:
                continue
            group_start = current_pos
            for (driver, data_arr, stat) in sorted(entries, key=lambda x: x[0]):
                positions.append(current_pos)
                last_name = driver.split("_")[-1]
                x_labels.append(f"{last_name}\n({stat[2]})")
                data_arrays.append(data_arr)
                annotations.append(f"{stat[0]} laps,\n avg: {stat[1]:.2f}s")
                current_pos += 1
            group_boundaries.append(((group_start + current_pos - 1) / 2.0, stint))
            current_pos += group_gap
        
        # Create the violin plot.
        violinplot = ax.violinplot(data_arrays, positions=positions, showmeans=True)
        driver_order = []
        for stint in sorted(group_data.keys()):
            entries = group_data[stint].get(metric, [])
            if len(entries) < 2:
                continue
            for (driver, _, _) in sorted(entries, key=lambda x: x[0]):
                driver_order.append(driver)
        for idx, body in enumerate(violinplot['bodies']):
            body.set_facecolor(driver_colors.get(driver_order[idx], "grey"))
            body.set_edgecolor('black')
            body.set_alpha(0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        for pos, data_arr, annot in zip(positions, data_arrays, annotations):
            local_max = data_arr.max()
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            offset = 0.05 * y_range
            y_annot = min(local_max + offset, ax.get_ylim()[1] - offset)
            ax.text(pos, y_annot, annot, rotation=0, ha='center', va='bottom', fontsize=8)
        
        
        ax.set_xlabel("Drivers")
        ax.set_ylabel(f"{metric} (seconds)")
        for center, stint in group_boundaries:
            ax.text(center, ax.get_ylim()[1], f"Stint {stint}", ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color='blue')
    
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    legend_handles = [mpatches.Patch(color=driver_colors.get(driver, "grey"), label=driver.replace("_", " "))
                      for driver in driver_names]
    fig.legend(handles=legend_handles, title="Drivers", loc='upper right', bbox_to_anchor=(0.97, 0.97))
    
    return fig

def plot_session_pace_laptimes(csv_data):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # prepare a color mapping for drivers
    driver_names = sorted(csv_data.keys())
    cmap = plt.get_cmap("tab10")
    driver_colors = {driver: cmap(i % 10) for i, driver in enumerate(driver_names)}
    
    for driver, df in csv_data.items():
        df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce")
        df = df.sort_values("LapNumber")
        
        lap_times = pd.to_timedelta(df["LapTime"].astype(str).str.strip(), errors="coerce").dropna()
        lap_seconds = lap_times.dt.total_seconds()
        laps = df["LapNumber"].loc[lap_seconds.index]
        
       
        if "PitOutTime" in df.columns or "PitInTime" in df.columns:
            pit_mask = df["PitOutTime"].astype(str).str.strip().apply(lambda x: x not in ["", "nan", "NaT"])
            pit_mask = pit_mask | df["PitInTime"].astype(str).str.strip().apply(lambda x: x not in ["", "nan", "NaT"])
            pit_mask = pit_mask.loc[lap_seconds.index]
        else:
            pit_mask = pd.Series(False, index=lap_seconds.index)
        
        # compare the drivers laps using 10 laps around the lap if possible
        outlier_mask = pd.Series(False, index=lap_seconds.index)
        n = len(lap_seconds)
        for i in range(n):
            # skip laps when driver pitted
            if pit_mask.iloc[i]:
                continue
            # 10 lap window
            start = max(0, i - 5)
            end = min(n, i + 6)
            # not current lap to compare
            neighbor_indices = list(range(start, i)) + list(range(i + 1, end))
            if not neighbor_indices:
                continue
            neighbors = lap_seconds.iloc[neighbor_indices]
            median_neighbor = neighbors.median()
            # use mad to get the median
            mad = np.median(np.abs(neighbors - median_neighbor))
            if mad == 0:
                mad = 0.1  
            if abs(lap_seconds.iloc[i] - median_neighbor) > 10 * mad:
                outlier_mask.iloc[i] = True
        
        line_lap_seconds = lap_seconds.copy()
        line_lap_seconds[pit_mask | outlier_mask] = np.nan
        
        color = driver_colors.get(driver, "grey")
        
        ax.plot(laps, line_lap_seconds, marker="o", linestyle="-", color=color, label=driver)
        
        pit_laps = laps[pit_mask]
        pit_times = lap_seconds[pit_mask]
        if not pit_laps.empty:
            ax.scatter(pit_laps, pit_times, marker="x", color=color, s=100, zorder=5)
        
        outlier_laps = laps[outlier_mask]
        outlier_times = lap_seconds[outlier_mask]
        if not outlier_laps.empty:
            ax.scatter(outlier_laps, outlier_times, marker="o", facecolors="none", edgecolors=color, s=100, zorder=5)
    
    ax.plot([], [], "o", color="black", label="Outlier")
    ax.plot([], [], "x", color="black", label="Pit Stop")
    ax.set_xlabel("Lap Number")
    ax.set_ylabel("Lap Time (seconds)")
    ax.set_title("Session Pace: Lap-by-Lap Trend")
    ax.legend()
    
    return fig

# Pre processing the lap data to analyse the tyres specifically
def prepare_driver_data_for_tyre_analysis(csv_data):
    processed_data = {}

    # lis tof columns needed for tyres
    cols_to_keep = [
        "Time", "LapNumber", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
        "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST", "TyreLife", "Stint", "Compound"
    ]
    
    for driver, df in csv_data.items():
        df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce")
        df = df.sort_values("LapNumber").reset_index(drop=True)
        
        # change lap times to seconds
        lap_times = pd.to_timedelta(df["LapTime"].astype(str).str.strip(), errors="coerce")
        lap_seconds = lap_times.dt.total_seconds()
        
        # get the pit stop laps
        if "PitOutTime" in df.columns or "PitInTime" in df.columns:
            pit_mask = df.get("PitOutTime", pd.Series("", index=df.index)).astype(str).str.strip().apply(
                lambda x: x not in ["", "nan", "NaT"])
            pit_mask = pit_mask | df.get("PitInTime", pd.Series("", index=df.index)).astype(str).str.strip().apply(
                lambda x: x not in ["", "nan", "NaT"])
        else:
            pit_mask = pd.Series(False, index=df.index)
        
        
        n = len(lap_seconds)
        outlier_mask = pd.Series(False, index=df.index)
        
        # use 10 lap window
        for i in range(n):
            if pit_mask.iloc[i]:
                continue  
            
            start = max(0, i - 5)
            end = min(n, i + 6)  
            
            neighbor_indices = list(range(start, i)) + list(range(i + 1, end))
            if not neighbor_indices:
                continue
            
            neighbors = lap_seconds.iloc[neighbor_indices]
            median_neighbor = neighbors.median()
            # find median
            mad = np.median(np.abs(neighbors - median_neighbor))
            if mad == 0:
                mad = 0.1 
            
            if abs(lap_seconds.iloc[i] - median_neighbor) > 10 * mad:
                outlier_mask.iloc[i] = True

        df_clean = df[~outlier_mask].copy()
        
        df_clean = df_clean[[col for col in cols_to_keep if col in df_clean.columns]]
        
        # sort by stint
        if "Stint" in df_clean.columns:
            df_clean = df_clean.sort_values("Stint").reset_index(drop=True)
        
        # get fuel percentage from race laps
        total_laps = len(df_clean)
        if total_laps > 0:
            # need atleast 1 percent remaning at end for cool down lap
            fuel_per_lap = 99 / total_laps
            fuel_percentages = []
            for lap in range(1, total_laps + 1):
                remaining = 100 - lap * fuel_per_lap
                
                if remaining < 1:
                    remaining = 1
                fuel_percentages.append(remaining)
            df_clean["FuelPercent"] = fuel_percentages
        
        processed_data[driver] = df_clean

    return processed_data

# get the weather data and preprocess by only getting weather data collected during the actual race
def get_weather_data(processed_driver_data, settings):
    
    base_weather_folder = "External_Data"
    year = settings.get("Year")
    race_event = settings.get("Race Event")
    
    if not year or not race_event:
        raise ValueError("Both 'Year' and 'Race Event' must be provided in the settings dictionary.")
    
    
    race_event_for_file = race_event.replace("_", " ")
    
    
    weather_csv_file = f"{year}_{race_event_for_file}_Weather_Data.csv"
    weather_csv_path = os.path.join(base_weather_folder, weather_csv_file)
    
    # read the file
    weather_df = pd.read_csv(weather_csv_path)
    
    # only race weather laps
    weather_df = weather_df[weather_df["Session"] == "R"].copy()
    
    weather_df["Time"] = pd.to_timedelta(weather_df["Time"], errors='coerce')
    
    # get time range for weather
    global_min_time = None
    global_max_time = None
    
    for df in processed_driver_data.values():
        if "Time" in df.columns:
            times = pd.to_timedelta(df["Time"], errors='coerce').dropna()
            if times.empty:
                continue
            min_time = times.min()
            max_time = times.max()
            if global_min_time is None or min_time < global_min_time:
                global_min_time = min_time
            if global_max_time is None or max_time > global_max_time:
                global_max_time = max_time
                
    if global_min_time is None or global_max_time is None:
        print("Warning: Could not determine a global time range from the driver data. Returning session 'R' data only.")
        return weather_df
    
    # filter weather for only during the race
    weather_df = weather_df[
        (weather_df["Time"] >= global_min_time) & (weather_df["Time"] <= global_max_time)
    ]
    print(weather_df)
    return weather_df

# combine to 1 dataframe for easy analysis
def combine_weather_and_driver_data(processed_driver_data, weather_df):
    
   # use time to combine
    weather_df = weather_df.copy()
    weather_df["Time"] = pd.to_timedelta(weather_df["Time"], errors='coerce')
    weather_df = weather_df.dropna(subset=["Time"])
    weather_df.sort_values("Time", inplace=True)
    
    merged_driver_data = {}
    
    for driver, df in processed_driver_data.items():
        df = df.copy()
       # convert the time
        df["Time"] = pd.to_timedelta(df["Time"], errors="coerce")
        df = df.dropna(subset=["Time"])
        df.sort_values("Time", inplace=True)
        
        # use one hot compound
        if "Compound" in df.columns:
            compound_dummies = pd.get_dummies(df["Compound"], prefix="Compound", drop_first=True)
            df = pd.concat([df.drop("Compound", axis=1), compound_dummies], axis=1)
        
       # merge based on closest time possible
        merged_df = pd.merge_asof(df, weather_df, on="Time", direction="nearest")
        merged_driver_data[driver] = merged_df
        
    return merged_driver_data

# regression analysis based on time
def regression_analysis_all_drivers_rf_with_features(merged_driver_data, target="LapTime"):
    
    # use merged data frame to get the drivers data
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    drivers = list(merged_driver_data.keys())
    cmap = plt.get_cmap("tab10")
    colors = {driver: cmap(i % 10) for i, driver in enumerate(drivers)}
    metrics_results = {}
    # create more data columns based on data for analysis
    for driver in drivers:
        df = merged_driver_data[driver].copy()
        if "Stint" in df.columns:
            df = df.groupby("Stint", group_keys=False).apply(lambda g: g.iloc[1:])
        df["LapTimeSeconds"] = pd.to_timedelta(df[target].astype(str), errors="coerce").dt.total_seconds()
        df.dropna(subset=["LapTimeSeconds"], inplace=True)
        if "Stint" in df.columns:
            df["Lag1"] = df.groupby("Stint")["LapTimeSeconds"].shift(1)
            df["Lag2"] = df.groupby("Stint")["LapTimeSeconds"].shift(2)
            df["NormLap"] = df.groupby("Stint")["LapNumber"].transform(lambda x: x / x.max())
        else:
            df["Lag1"] = df["LapTimeSeconds"].shift(1)
            df["Lag2"] = df["LapTimeSeconds"].shift(2)
            df["NormLap"] = df["LapNumber"] / df["LapNumber"].max()
        if "TyreLife" in df.columns and "FuelPercent" in df.columns:
            df["TyreLife_FuelPercent"] = df["TyreLife"] * df["FuelPercent"]
        if "FuelPercent" in df.columns and "NormLap" in df.columns:
            df["FuelPercent_LapNumber"] = df["FuelPercent"] * df["NormLap"]
        if "Stint" in df.columns and "Compound" in df.columns:
            df["StintCompound"] = df.groupby("Stint")["Compound"].transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
            dummies = pd.get_dummies(df["StintCompound"], prefix="Compound", drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            # calculate more data based on the data recieved
        base = ["TyreLife","NormLap","FuelPercent","TrackTemp","Humidity","Pressure","AirTemp","Lag1","Lag2","TyreLife_FuelPercent","FuelPercent_LapNumber"]
        comps = [c for c in df.columns if c.startswith("Compound_")]
        feats = base + comps
        df.dropna(subset=feats, inplace=True)
        if df.empty:
            continue
        X = df[feats]
        y = df["LapTimeSeconds"]
        # scale and then do random forest to do regression analysis
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        model = RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_split=3, min_samples_leaf=2, random_state=42, oob_score=True)
        y_pred = cross_val_predict(model, Xs, y, cv=kf)
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        model.fit(Xs, y)
        oob = model.oob_score_
        # showcase metrics for analysis
        metrics_results[driver] = {"RF": {"MAE": mae, "MSE": mse, "R2": r2, "OOB": oob}}
        axs[0].scatter(y, y_pred, alpha=0.7, edgecolors="k", color=colors[driver], label=driver)
        res = y - y_pred
        axs[1].scatter(y_pred, res, alpha=0.7, edgecolors="k", color=colors[driver], label=driver)
    
    # show both graphs based on the data 
    axs[0].set_title("Actual vs Predicted Lap Time")
    axs[0].set_xlabel("Actual Lap Time (s)")
    axs[0].set_ylabel("Predicted Lap Time (s)")
    mn = min(axs[0].get_xlim()[0], axs[0].get_ylim()[0])
    mx = max(axs[0].get_xlim()[1], axs[0].get_ylim()[1])
    axs[0].plot([mn, mx], [mn, mx], color="red", linestyle="--")
    axs[1].set_title("Residual Plot")
    axs[1].set_xlabel("Predicted Lap Time (s)")
    axs[1].set_ylabel("Residuals (s)")
    h, l = axs[0].get_legend_handles_labels()
    d = dict(zip(l, h))
    axs[0].legend(d.values(), d.keys(), title="Driver")
    plt.tight_layout()
    plt.show()
    return metrics_results, [fig]


# get optimal pit stop timing based on degredation of tyres using laptimes
def find_optimal_pit_stop_tree(merged_driver_data, window_size=3, lap_time_increase_threshold=1.0, min_samples=5):
    
    recommendations = {}
    for driver, df in merged_driver_data.items():
        if 'LapNumber' not in df.columns or 'Stint' not in df.columns or 'LapTime' not in df.columns:
            continue
        df = df.sort_values("LapNumber").reset_index(drop=True)
        driver_recs = []
        for stint in df["Stint"].unique():
            stint_df = df[df["Stint"] == stint].copy().reset_index(drop=True)
            if len(stint_df) < window_size + 1:
                continue
            stint_df["LapTimeSeconds"] = pd.to_timedelta(
                stint_df["LapTime"].astype(str).str.strip(), errors="coerce"
            ).dt.total_seconds()
            if stint_df["LapTimeSeconds"].isna().all():
                continue
            n_laps = len(stint_df)
            # get the fuel per laps used
            stint_df["FuelPercent"] = [max(100 - (lap * (99 / n_laps)), 1) for lap in range(1, n_laps + 1)]
            features_list = []
            target_list = []
            lap_numbers = []
            for i in range(window_size, len(stint_df)):
                rolling_mean_time = stint_df["LapTimeSeconds"].iloc[i - window_size:i].mean()
                lap_time_diff = stint_df["LapTimeSeconds"].iloc[i] - rolling_mean_time
                norm_lap = stint_df["LapNumber"].iloc[i] / stint_df["LapNumber"].max()
                lap_trend = (stint_df["LapTimeSeconds"].iloc[i] - stint_df["LapTimeSeconds"].iloc[i - window_size]) / window_size
                feature_dict = {
                    "norm_lap": norm_lap,
                    "rolling_mean_time": rolling_mean_time,
                    "current_lap_time": stint_df["LapTimeSeconds"].iloc[i],
                    "lap_time_diff": lap_time_diff,
                    "lap_trend": lap_trend
                }
                # calculate degredation based on external weather factors and car speeds per lap difference
                for col in ["TrackTemp", "Pressure", "AirTemp", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]:
                    if col in stint_df.columns:
                        rolling_mean = stint_df[col].iloc[i - window_size:i].mean()
                        feature_dict[col + "_diff"] = stint_df[col].iloc[i] - rolling_mean
                if "Rainfall" in stint_df.columns:
                    rolling_mean = stint_df["Rainfall"].iloc[i - window_size:i].astype(int).mean()
                    feature_dict["Rainfall_diff"] = int(stint_df["Rainfall"].iloc[i]) - rolling_mean
                if "FuelPercent" in stint_df.columns:
                    rolling_mean = pd.Series(stint_df["FuelPercent"].iloc[i - window_size:i]).mean()
                    feature_dict["FuelPercent_diff"] = stint_df["FuelPercent"].iloc[i] - rolling_mean
                features_list.append(feature_dict)
                target_list.append(lap_time_diff)
                lap_numbers.append(stint_df["LapNumber"].iloc[i])
            features_df = pd.DataFrame(features_list)
            target_series = pd.Series(target_list)

            
            features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            target_series.replace([np.inf, -np.inf], np.nan, inplace=True)

            # fill missing with median values so this doesnt break
            features_df.fillna(features_df.median(), inplace=True)
            target_series.fillna(target_series.median(), inplace=True)

            if len(features_df) < min_samples:
                continue
                
            param_grid = {"max_depth": [3, 5, 7, None], "min_samples_split": [2, 5, 10]}
            grid = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=3)
            grid.fit(features_df, target_series)
            model = grid.best_estimator_
            predictions = model.predict(features_df)
            max_idx = predictions.argmax()
            best_candidate = predictions[max_idx]
            recommended_lap = lap_numbers[max_idx]
            if best_candidate > lap_time_increase_threshold:
                rec = {
                    "Stint": stint,
                    "RecommendedPitLap": recommended_lap,
                    "PredictedDegradation": best_candidate
                }
                actual_pit_lap = None
                compound_changed_to = None
                if "Compound" in stint_df.columns:
                    compounds = stint_df["Compound"].tolist()
                    for j in range(1, len(compounds)):
                        if compounds[j] != compounds[j - 1]:
                            actual_pit_lap = stint_df.loc[j, "LapNumber"]
                            compound_changed_to = compounds[j]
                            break
                if actual_pit_lap is not None:
                    rec["ActualPitLap"] = actual_pit_lap
                    rec["CompoundChangedTo"] = compound_changed_to
                    rec["LapDifference"] = actual_pit_lap - recommended_lap
                driver_recs.append(rec)
        recommendations[driver] = driver_recs
        
    return recommendations






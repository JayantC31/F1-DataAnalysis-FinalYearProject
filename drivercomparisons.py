import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def getDataFromCsv(settings):
    """
    Run the driver analysis based on the settings that is given from the user
    
    """
    # get the data into multiple dataframes
    base_folder = "Telemetry_Data"
    year = settings.get("Year")
    race_event = settings.get("Race Event")
    drivers = settings.get("Drivers", [])
    session_file = settings.get("Session Summary File")
    lap_number = settings.get("Lap Number")
    CSVDATA = {}
    # check if session if qualifying or the race
    if session_file.startswith("Q"):
        session_folder = "Q"
    elif session_file.startswith("R"):
        session_folder = "R"
    else:
        print("Session type not recognized. Skipping analysis.")
        return CSVDATA
    # check for each driver
    for driver in drivers:
        # get the path for the driver csv
        driver_path = os.path.join(base_folder, year, driver, race_event)
        telemetry_csv = os.path.join(driver_path, session_folder, "Telemetry", f"Lap_{float(lap_number)}_telemetry.csv")
        if not os.path.exists(telemetry_csv):
            print(f"Telemetry CSV not found for {driver} at {telemetry_csv}")
            telemetry_data = pd.DataFrame() # empty csv
        else:
            telemetry_data = pd.read_csv(telemetry_csv)
        # get the data for the driver
        CSVDATA[driver] = {
            "telemetry": telemetry_data
        }
    
    return CSVDATA

# process data before the graphs can be made considering theres like 5 graphs so need to process
def BeforeGRAPHS(driver_data):
    """
    Get the graphs for the drivers raw data
    """
    # need 5 graphs
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    for driver, data in driver_data.items():
        telemetry_df = data.get("telemetry")
        # check if the data is valid
        # get the data for the driver based on distance
        if telemetry_df is None or telemetry_df.empty or "Distance" not in telemetry_df.columns:
            continue
        telemetry_df.sort_values("Distance", inplace=True)
        if "nGear_x" in telemetry_df.columns:
            # get the gear data make sure it is between 1 and 8
            valid_geardata = telemetry_df[(telemetry_df["nGear_x"] >= 1) & (telemetry_df["nGear_x"] <= 8)].copy()
            valid_geardata.sort_values("Distance", inplace=True)
            ax1.plot(valid_geardata["Distance"], valid_geardata["nGear_x"], label=driver, linewidth=1.0)
        if "Brake_x" in telemetry_df.columns:
            ax2.plot(telemetry_df["Distance"], telemetry_df["Brake_x"], label=driver, linewidth=1.0)
        if "Throttle_x" in telemetry_df.columns:
            ax3.plot(telemetry_df["Distance"], telemetry_df["Throttle_x"], label=driver, linewidth=1.0)
        if "Speed_x" in telemetry_df.columns:
            ax4.plot(telemetry_df["Distance"], telemetry_df["Speed_x"], label=driver, linewidth=1.0)
        if "RPM_x" in telemetry_df.columns:
            ax5.plot(telemetry_df["Distance"], telemetry_df["RPM_x"], label=driver, linewidth=1.0)
    # set the labels for the graphs
    ax1.set_ylabel("Gear (-)")
    ax1.set_yticks([2, 4, 6, 8])
    ax1.set_title("Driver telemetry comparisons")
    ax2.set_ylabel("Brake usage")
    ax2.set_yticks([0, 1])
    ax2.set_ylim(0, 1.01)
    ax3.set_ylabel("Throttle, pedal pressure [%]")
    ax4.set_ylabel("Car Speed [km/h]")
    ax5.set_ylabel("RPM")
    ax5.set_xlabel("Distance [metres]")
    handles, labels = ax1.get_legend_handles_labels()
    # make the legend on the topright
    fig.subplots_adjust(top=0.90)
    fig.legend(handles, labels, title="Drivers", loc='upper right', bbox_to_anchor=(0.95, 0.95))
    return fig

def align_telemetry_data(driver_data, num_points=500):
    """
    Align the telemetry data for the drivers
    This is needed cayse there is an uneven amount of rows for telemetry for each driver
    Also, distance and laptime covered is different for each driver
    """
    # 500 seems to be the most amount of rows ive seen in the files
    # get the min and max distance for the drivers
    min_distance, max_distance = None, None
    for driver, data in driver_data.items():
        telemetry_df = data.get("telemetry")
        # check that the distance is in the columns and it aint empty
        if telemetry_df is None or telemetry_df.empty or "Distance" not in telemetry_df.columns:
            continue
        telemetry_df["Distance"] = pd.to_numeric(telemetry_df["Distance"], errors="coerce")
        # find the min and max distance
        dmin, dmax = telemetry_df["Distance"].min(), telemetry_df["Distance"].max()
        min_distance = dmin if min_distance is None else max(min_distance, dmin)
        max_distance = dmax if max_distance is None else min(max_distance, dmax)
    
    if min_distance is None or max_distance is None or min_distance >= max_distance:
        print("the distance data is not valid")
        return driver_data
    # find the common amount of distances for the drivers
    common_distance = np.linspace(min_distance, max_distance, num_points)
    aligned_driver_data = {}
    for driver, data in driver_data.items():
        telemetry_df = data.get("telemetry")
        if telemetry_df is None or telemetry_df.empty or "Distance" not in telemetry_df.columns:
            aligned_driver_data[driver] = data
            continue
        # align the data for the driver
        telemetry_df["Distance"] = pd.to_numeric(telemetry_df["Distance"], errors="coerce")
        aligned_df = pd.DataFrame({"Distance": common_distance})
        for eachcoloumn in telemetry_df.columns:
            if eachcoloumn == "Distance":
                continue
            # get the data for the column
            col_data = pd.to_numeric(telemetry_df[eachcoloumn], errors="coerce")
            df_sorted = telemetry_df.copy()
            df_sorted[eachcoloumn] = col_data
            df_sorted.sort_values("Distance", inplace=True)
            # interpolate the data by distance
            aligned_df[eachcoloumn] = np.interp(common_distance, df_sorted["Distance"], df_sorted[eachcoloumn])
        new_data = data.copy()
        new_data["telemetry"] = aligned_df
        aligned_driver_data[driver] = new_data
    # telemetry data now aligned to the common distance points
    return aligned_driver_data

def extract_features(driver_data):
    """
    This is needed to get all the features that i can use to compare the drivers and their driving styles
    """
    # get the list and drivers
    features_list = []
    driver_names = []
    for driver, data in driver_data.items():
        telemetry_df = data.get("telemetry")
        if telemetry_df is None or telemetry_df.empty:
            continue
        feature_dict = {} # make a dictionary for the features that i need
        # gear Features
        if "nGear_x" in telemetry_df.columns:
            gear_series = telemetry_df["nGear_x"]
            # make sure gears are between 1 and 8
            valid_gears = gear_series[(gear_series >= 1) & (gear_series <= 8)]
            # average gear and gear frequency of shifts to show how many times it changes
            feature_dict["avg_gear"] = valid_gears.mean() if not valid_gears.empty else np.nan
            feature_dict["gear_shift_freq"] = np.sum(valid_gears.diff().fillna(0) != 0)
            total_points = len(valid_gears)
            for g in range(1, 9): # how much time is spent in each gear
                feature_dict[f"time_in_gear_{g}"] = np.sum(valid_gears == g) / total_points if total_points > 0 else np.nan
        else: # if no gear data then assume they all equal to nan
            feature_dict["avg_gear"] = feature_dict["gear_shift_freq"] = np.nan
            for g in range(1, 9):
                feature_dict[f"time_in_gear_{g}"] = np.nan
        
        # brake Features
        if "Brake_x" in telemetry_df.columns:
            brake_series = telemetry_df["Brake_x"]
            feature_dict["brake_usage_mean"] = brake_series.mean()
            brake_binary = (brake_series > 0).astype(int)
            feature_dict["brake_freq"] = np.sum(brake_binary.diff().fillna(0) == 1)
            # Calculate the number of points where braking is True
        else:
            feature_dict["brake_usage_mean"] = feature_dict["brake_freq"] =  np.nan
        
        # throttle Features
        if "Throttle_x" in telemetry_df.columns and "Distance" in telemetry_df.columns:
            throttle_series = telemetry_df["Throttle_x"]
            feature_dict["avg_throttle"] = throttle_series.mean()
            distance = pd.to_numeric(telemetry_df["Distance"], errors="coerce").values
            throttle = throttle_series.values
            diff_distance = np.diff(distance)
            diff_throttle = np.diff(throttle)
            valid = diff_distance != 0
            # i want to measure if a driver is smooth or more aggressive with the throttle and how often they are full throttle as a fraction
            # this shows the pressure a driver has on throttle compared to 0 or 100% throttle
            feature_dict["throttle_response_rate"] = np.mean(np.abs(diff_throttle[valid] / diff_distance[valid])) if np.sum(valid) > 0 else np.nan
            feature_dict["frac_full_throttle"] = np.mean(throttle_series >= 0.99)
        else:
            feature_dict["avg_throttle"] = feature_dict["throttle_response_rate"] = feature_dict["frac_full_throttle"] = np.nan
        
        # speed Features
        if "Speed_x" in telemetry_df.columns and "Distance" in telemetry_df.columns:
            speed_series = telemetry_df["Speed_x"]
            feature_dict["avg_speed"] = speed_series.mean()
            feature_dict["max_speed"] = speed_series.max()
            distance = pd.to_numeric(telemetry_df["Distance"], errors="coerce").values
            speed = speed_series.values
            diff_distance = np.diff(distance)
            diff_speed = np.diff(speed)
            valid = diff_distance != 0
            if np.sum(valid) > 0:
                acceleration = diff_speed[valid] / diff_distance[valid]
                # measure accleartation and deceleration
                feature_dict["avg_acceleration"] = np.mean(acceleration[acceleration > 0]) if np.any(acceleration > 0) else 0
                feature_dict["avg_deceleration"] = np.mean(acceleration[acceleration < 0]) if np.any(acceleration < 0) else 0
            else:
                feature_dict["avg_acceleration"] = feature_dict["avg_deceleration"] = np.nan
        else:
            feature_dict["avg_speed"] = feature_dict["max_speed"] = feature_dict["avg_acceleration"] = feature_dict["avg_deceleration"] = np.nan
        
        # RPM Features
        if "RPM_x" in telemetry_df.columns and "Distance" in telemetry_df.columns:
            rpm_series = telemetry_df["RPM_x"]
            feature_dict["avg_rpm"] = rpm_series.mean()
            # find average rpm and the rate of change of rpm
            distance = pd.to_numeric(telemetry_df["Distance"], errors="coerce").values
            rpm = rpm_series.values
            diff_distance = np.diff(distance)
            diff_rpm = np.diff(rpm)
            valid = diff_distance != 0
            if np.sum(valid) > 0:
                rpm_rate = diff_rpm[valid] / diff_distance[valid]
                # average of all posivite rates of change of rpm to show how fast the driver can accelerate the engine and also show the engines response rate
                # ramp down is the opposite
                feature_dict["rpm_ramp_up_rate"] = np.mean(rpm_rate[rpm_rate > 0]) if np.any(rpm_rate > 0) else 0
                feature_dict["rpm_ramp_down_rate"] = np.mean(rpm_rate[rpm_rate < 0]) if np.any(rpm_rate < 0) else 0
            else:
                feature_dict["rpm_ramp_up_rate"] = feature_dict["rpm_ramp_down_rate"] = np.nan
        else:
            feature_dict["avg_rpm"] = feature_dict["rpm_ramp_up_rate"] = feature_dict["rpm_ramp_down_rate"] = np.nan
        


        # add them all to the list per drivers
        driver_names.append(driver)
        features_list.append(feature_dict)
    
    return pd.DataFrame(features_list, index=driver_names)

def scale_features(featuresdata):
    """
    needed to scale the features to make sure they are all on the same scale
    some features like rpm and speed are on different scales"""
    FilledDATAFRAME = featuresdata.fillna(featuresdata.mean())
    scaler = StandardScaler()
    # use the standard scaler to scale the features
    scaled_features = scaler.fit_transform(FilledDATAFRAME)
    return scaled_features, scaler

def PcaAnalysis(driver_data):
    """
    this gets the pca analysis for the drivers based on the features
    """
    featuresdata = extract_features(driver_data)
    
    # Scale the features using the scaler function
    scaled_features, scaler = scale_features(featuresdata)
    
    # Run PCA to reduce to 2 dimensions with random state to stop random
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(scaled_features)
    
    # plot of the pca results
    fig_pca, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], cmap='viridis', alpha=0.7)
    
    # annotate the drivers on the plot
    for i, driver in enumerate(featuresdata.index):
        ax.annotate(driver, (pca_result[i, 0], pca_result[i, 1]), fontsize=8)
    
    ax.set_xlabel(" Component 1")
    ax.set_ylabel(" Component 2")
    ax.set_title("PCA Analysis of Driver Telemetry Features")
    ax.grid(True)
    #plt.show()
    
    # get the explained variance ratio
    variance_ratio = pca.explained_variance_ratio_
    
    # table of the features for components
    feature_table = pd.DataFrame(
        pca.components_.T,
        index=featuresdata.columns,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)]
    )
    
    return fig_pca, variance_ratio, feature_table

def get_top_features_for_pcs(feature_table, top_n=5):
    """
    get the most important features for each PCA component
    """
    rows = []
    for comp in feature_table.columns:
        # Sort features by loading for current component (highest first)
        sorted_features = feature_table[comp].sort_values(ascending=False)
        # Top positive loadings (High)
        for feature, loading in sorted_features.head(top_n).items():
            rows.append({
                'PCA Component': comp,
                'Direction': 'High',
                'Feature': feature,
                'Loading': loading
            })
        # Top negative loadings (Lowest values)
        sorted_features_low = feature_table[comp].sort_values(ascending=True)
        for feature, loading in sorted_features_low.head(top_n).items():
            rows.append({
                'PCA Component': comp,
                'Direction': 'Low',
                'Feature': feature,
                'Loading': loading
            })
    return pd.DataFrame(rows)




def plot_radar_chart_scaled(featuresdata):
    # hget the scaler to scale the features for the chart
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(featuresdata)
    scaled_df = pd.DataFrame(scaled_values, index=featuresdata.index, columns=featuresdata.columns)
    # list the categories for the radar chart
    categories = list(scaled_df.columns)
    catlength = len(categories)
    # get the angles for the chart
    angles = [n / float(catlength) * 2 * np.pi for n in range(catlength)]
    angles += angles[:1]
    # plot as square
    plt.figure(figsize=(8,8))
    ax = plt.subplot(111, polar=True)
    # for each driver plot the points per category
    for driver, row in scaled_df.iterrows():
        values = row.values.tolist()
        # plot the first value to the end to make it a closed loop
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2, label=driver)
        ax.fill(angles, values, alpha=0.25)
    plt.xticks(angles[:-1], categories)
    # set the position of the labels
    ax.set_rlabel_position(30)
    plt.yticks([-2, -1, 0, 1, 2], ["-2", "-1", "0", "1", "2"])
    plt.title("Radar Chart of Scaled Driver Features", y=1.1)
    # make legend top right
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.show()

def plot_time_series_differences(aligned_data, metric):
    """
    plot the difference per distance for the driver but based on the metric
    """
    driver_values = {}
    for driver, d in aligned_data.items():
        telemetry_df = d.get("telemetry") # open the telemetry data based on the metric found in the columns of the data
        if telemetry_df is not None and not telemetry_df.empty and metric in telemetry_df.columns:
            driver_values[driver] = telemetry_df[metric].values
    if not driver_values:
        print("No valid data found for metric:", metric)
        return None
    # get the average metric for all drivers
    drivers = list(driver_values.keys())
    all_values = np.array(list(driver_values.values()))
    avg_metric = np.mean(all_values, axis=0)
    # plot the difference per distance for each driver
    fig, ax = plt.subplots(figsize=(10,6))
    for driver in drivers:
        diff = driver_values[driver] - avg_metric
        # mark the differences for each driver
        distance = aligned_data[driver]["telemetry"]["Distance"].values
        ax.plot(distance, diff, label=driver)
    #ax.setgrid(True)
    ax.set_title(f"Difference in {metric} Relative to Average")
    ax.set_xlabel("Distance (metres)")
    # fill the area between the lines
    #ax.fill(distance, np.zeros_like(distance), 'k', alpha=0.1)
    ax.set_ylabel(f"Difference in {metric}")
    ax.legend()
    
    return fig

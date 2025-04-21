import os
import pandas as pd
import streamlit as st
import TrackStats as ts
import matplotlib.pyplot as plt
from io import BytesIO
import drivercomparisons as dc
import strategies as stg
from streamlit_option_menu import option_menu
from PIL import Image
# Function to display the CSV file
def display_csv(file_path):
    """
    display the csv on screen 
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        # Load and display the CSV file (showing up to 100000 rows)
        if df is not None:
            st.write("### Data Showcase")
            st.write(df.head(100000))
    except Exception as e:
        # Show any exceptions on screen
        st.error(f"Error loading the file: {e}")

def telemetry_menu():
    """ get telemetry meny for telemetry data"""
    base_folder = "Telemetry_Data"
    
    # ask for year
    year_options = sorted([d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))])
    if not year_options:
        st.error("No year folders found in Telemetry_Data!")
        return
    selected_year = st.selectbox("Select a Year", year_options)
    year_folder = os.path.join(base_folder, selected_year)
    
    # Now, select the driver from within the selected year's folder
    drivers = sorted([d for d in os.listdir(year_folder) if os.path.isdir(os.path.join(year_folder, d))])
    if not drivers:
        st.error("There are no drivers found for the selected year.")
        return
    driver_choice = st.selectbox("Select a driver", drivers)
    driver_path = os.path.join(year_folder, driver_choice)

    # select race
    races = sorted([d for d in os.listdir(driver_path) if os.path.isdir(os.path.join(driver_path, d))])
    if not races:
        st.error("No race folders found for this driver in the selected year.")
        return
    race_choice = st.selectbox("Select a race", races)
    race_path = os.path.join(driver_path, race_choice)

    # telemetry option
    telemetry_option = st.radio("Display telemetry data types:",
                                ("Session Summary", "Specific Lap Data"))

    if telemetry_option == "Session Summary":
        # list csv
        session_summary_files = [f for f in os.listdir(race_path) if f.endswith("_laps.csv")]
        if not session_summary_files:
            st.error("No session summary files found in this race folder.")
            return
        session_file_choice = st.selectbox("Select a Session Summary File", sorted(session_summary_files))
        file_path = os.path.join(race_path, session_file_choice)
        st.write(f"**Opening file:** {file_path}")
        display_csv(file_path)

    elif telemetry_option == "Specific Lap Data":
       
        session_folders = [d for d in os.listdir(race_path) if os.path.isdir(os.path.join(race_path, d))]
        if not session_folders:
            st.error("No session folders found in this race folder.")
            return
        session_choice = st.selectbox("Select a Session", sorted(session_folders))
        session_path = os.path.join(race_path, session_choice)
        telemetry_subfolder = os.path.join(session_path, "Telemetry")
        if not os.path.exists(telemetry_subfolder):
            st.error("Telemetry subfolder not found in the selected session.")
            return
        
        lap_files = [f for f in os.listdir(telemetry_subfolder) if f.endswith("_telemetry.csv")]
        if not lap_files:
            st.error("No lap telemetry files found in the selected session.")
            return
        lap_file_choice = st.selectbox("Select a Lap Telemetry File", sorted(lap_files))
        file_path = os.path.join(telemetry_subfolder, lap_file_choice)
        st.write(f"**Opening file:** {file_path}")
        display_csv(file_path)

def external_menu(data_type):
    """
    For viewing weather or tire data
    
    """
    base_folder = "External_Data"
    if not os.path.exists(base_folder):
        st.error("External_Data folder not found!")
        return

    if data_type.lower() == "weather":
        filter_str = "Weather_Data.csv"
    elif data_type.lower() == "tire":
        filter_str = "Tire_Data.csv"
    else:
        st.error("Invalid external data type")
        return

    # list files
    files = [f for f in os.listdir(base_folder)
             if os.path.isfile(os.path.join(base_folder, f)) and filter_str in f]
    if not files:
        st.error(f"No files found for {data_type} data")
        return

    # get files by year
    years = {}
    for f in files:
        parts = f.split("_")
        if parts:
            year = parts[0]
            years.setdefault(year, []).append(f)

    year_list = sorted(years.keys())
    selected_year = st.selectbox("Select a Year", year_list)
    files_for_year = sorted(years[selected_year])
    selected_file = st.selectbox("Select a File", files_for_year)
    file_path = os.path.join(base_folder, selected_file)
    st.write(f"**Opening file:** {file_path}")
    display_csv(file_path)

@st.cache_data
def getRaceData(season):
    """
    # get race data for track stats to collect all data easier
    
    """
    year = season
    lap_summaries = []
    session_telemetry_data = []
    base_dir = os.path.join("Telemetry_Data", str(year))
    
    # Loop through each driver
    for driver in os.listdir(base_dir):
        driver_path = os.path.join(base_dir, driver)
        if not os.path.isdir(driver_path):
            continue
        
        # Loop through each race folder 
        for race in os.listdir(driver_path):
            race_path = os.path.join(driver_path, race)
            if not os.path.isdir(race_path):
                continue
            
            # Check for the lap summary CSV
            laps_csv_path = os.path.join(race_path, "R_laps.csv")
            if not os.path.exists(laps_csv_path):
                continue
            
            # need these columns
            lap_columns = ['Time', 
                           'LapNumber', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 
                           'SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'TyreLife', 'Compound', 'LapStartTime'
                          ]
            try:
                lap_df = pd.read_csv(laps_csv_path, usecols=lap_columns)
                
                invalid_rows = lap_df[lap_df.isna().any(axis=1)]
                invalid_lap_numbers = set(invalid_rows['LapNumber'].tolist())
                lap_df = lap_df.dropna()
                lap_df["driver"] = driver
                lap_df["race"] = race
                lap_summaries.append(lap_df)
            except Exception as e:
                print(f"Error reading {laps_csv_path}: {e}")
                continue
            
            # get telemetry and clean
            telemetry_folder = os.path.join(race_path, "R", "Telemetry")
            if os.path.exists(telemetry_folder) and os.path.isdir(telemetry_folder):
                # sort by lap
                def extract_lap(file):
                    parts = file.split("_")
                    if len(parts) >= 2:
                        try:
                            return int(float(parts[1]))
                        except:
                            return float('inf')
                    return float('inf')
                
                telemetry_files = sorted(
                    [f for f in os.listdir(telemetry_folder) if f.endswith(".csv")],
                    key=extract_lap
                )
                for file in telemetry_files:
                    file_path = os.path.join(telemetry_folder, file)
                    try:
                        parts = file.split("_")
                        if len(parts) >= 2:
                            lap_num = int(float(parts[1]))
                        else:
                            lap_num = None
                        # Skip invalid laps.
                        if lap_num is not None and lap_num in invalid_lap_numbers:
                            continue
                        telemetry_columns = ['Time', 'RPM_x', 'Speed_x', 'Throttle_x', 'Brake_x', 
                                             'Distance', 'X', 'Y', 'Z', 'DRS_x', 'nGear_x' ]
                        tel_df = pd.read_csv(file_path, usecols=telemetry_columns)
                        tel_df["driver"] = driver
                        tel_df["race"] = race
                        tel_df["lap"] = lap_num
                        session_telemetry_data.append(tel_df)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
            else:
                print("No telemetry folder found for this race")
    
    lap_summary_df = pd.concat(lap_summaries, ignore_index=True) if lap_summaries else pd.DataFrame()
    telemetry_df = pd.concat(session_telemetry_data, ignore_index=True) if session_telemetry_data else pd.DataFrame()
    print(f"Collected data for {len(lap_summary_df)} laps and {len(telemetry_df)} telemetry records.")
    
    # Process weather data
    weather_data_list = []
    weather_base_dir = "External_Data"
    for file in os.listdir(weather_base_dir):
        if file.startswith(str(year) + "_") and "Weather_Data.csv" in file:
            file_path = os.path.join(weather_base_dir, file)
            try:
                # Read only these weather columns 
                weather_columns = ['AirTemp', 'Humidity', 'Pressure', 'Rainfall', 
                                   'TrackTemp', 'WindDirection', 'WindSpeed', 'Session']
                weather_df = pd.read_csv(file_path, usecols=weather_columns)
                # onlyt race
                weather_df = weather_df[weather_df["Session"] == "R"]
                race_name = file[len(str(season)) + 1 : file.rfind("_Weather_Data.csv")]
                weather_df["race"] = race_name
                weather_data_list.append(weather_df)
            except Exception as e:
                print(f"Error reading weather file {file_path}: {e}")
    
    weather_df = pd.concat(weather_data_list, ignore_index=True) if weather_data_list else pd.DataFrame()
    print(f"Collected weather data for {len(weather_df)} records.")
    
    return lap_summary_df, telemetry_df, weather_df

def CustomOptions(require_session=True, require_lap=True):
    # set flags to True when displaying options
    RaceOptionsNeeded = True
    SessionOptionsNeeded = require_session
    LapOptionNeeded = require_lap
    needmultidrivers = True

    settings = {}
    
    base_folder = "Telemetry_Data"
    if RaceOptionsNeeded:
        # Ask for the year first
        year_options = sorted([d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))])
        if not year_options:
            st.error("No year folders found in Telemetry_Data!")
            return None
        selected_year = st.selectbox("Select the Year", year_options)
        year_folder = os.path.join(base_folder, selected_year)
        
        # Select drivers
        drivers = sorted([d for d in os.listdir(year_folder) if os.path.isdir(os.path.join(year_folder, d))])
        if not drivers:
            st.error("There are no drivers found for the selected year.")
            return None
        driver_choice = st.multiselect("Select the drivers you want to compare to", drivers)
        
        if not driver_choice:
            st.write("Please select at least one driver")
            return None
        elif len(driver_choice) == 1 and needmultidrivers:
            st.write("Please select at least two drivers")
            return None
        
        # get race
        driver_path = os.path.join(year_folder, driver_choice[0])
        races = sorted([d for d in os.listdir(driver_path) if os.path.isdir(os.path.join(driver_path, d))])
        if not races:
            st.error("No races found for the selected driver.")
            return None
        race_choice = st.selectbox("Select the race in which the drivers participated", races)
        
        
        if SessionOptionsNeeded:
            race_path = os.path.join(driver_path, race_choice)
            session_summary_files = [f for f in os.listdir(race_path) if f.endswith("_laps.csv")]
            if not session_summary_files:
                st.error("No session summary files found in this race folder.")
                return None
            session_file_choice = st.selectbox("Select a Session", sorted(session_summary_files))
            settings["Session Summary File"] = session_file_choice
        
        
        if LapOptionNeeded:
            common_laps = None
            for driver in driver_choice:
                csv_path = os.path.join(year_folder, driver, race_choice, session_file_choice)
                try:
                    df = pd.read_csv(csv_path)
                    driver_laps = set(df["LapNumber"].unique())
                    if common_laps is None:
                        common_laps = driver_laps
                    else:
                        common_laps = common_laps.intersection(driver_laps)
                except Exception as e:
                    st.error(f"Error reading lap data for driver {driver}: {e}")
                    return None
            
            if common_laps and len(common_laps) > 0:
                common_laps = sorted(common_laps)
                selected_common_lap = st.selectbox("Select Common Lap Number", common_laps)
                settings["Lap Number"] = selected_common_lap
            else:
                st.error("No common lap number found for all selected drivers.")
                return None
        
        # store in settings
        settings["Year"] = selected_year
        settings["Race Event"] = race_choice
        settings["Drivers"] = driver_choice

    # button to run the analysis
    if st.button("Run Analysis"):
        st.write("Running analysis with the following settings:")
        st.write(settings)
        return settings
    else:
        return None

def main():
    
    st.markdown(
        """
        <style>
        /* Main Background */
        .stApp {
            background-color: #121212;
        }
        /* General Text Color */
        body, .stApp {
            color: #FFFFFF;
        }
        /* Headings and Accent Elements */
        h1, h2, h3, .custom-title {
            color: #FFFFFF;
        }
        
        /* --------------------------
        Sidebar
        -------------------------- */
        [data-testid="stSidebar"] {
            background-color: #1E1E1E;
        }
        [data-testid="stSidebar"] * {
            color: #FFFFFF;
        }
        /* Sidebar radio button*/
        [data-testid="stSidebar"] .stRadio > div {
            background-color: #FF8000;
            border-radius: 5px;
            padding: 5px;
        }
        [data-testid="stSidebar"] .stRadio label {
            color: #FFFFFF;
        }
        /* Sidebar select box */
        [data-testid="stSidebar"] div[data-baseweb="select"] {
            background-color: #1E1E1E !important;
            border: 1px solid #1E1E1E !important;
            color: #FFFFFF !important;
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background-color: #1E1E1E !important;
            color: #FFFFFF !important;
        }
        
        /* --------------------------
        Content 
        -------------------------- */
        /* Main content radio button styling (not in sidebar) */
        .main .block-container .stRadio > div {
            background-color: #333333;
            border-radius: 5px;
            padding: 5px;
        }
        .main .block-container .stRadio label {
            color: #FFFFFF;
        }
        /* Main content select box styling (if needed) */
        .main .block-container div[data-baseweb="select"] {
            background-color: #333333 !important;
            border: 1px solid #333333 !important;
            color: #FFFFFF !important;
        }
        .main .block-container div[data-baseweb="select"] > div {
            background-color: #333333 !important;
            color: #FFFFFF !important;
        }
        
        /* CSV Container */
        div[data-testid="stDataFrameContainer"] {
            background-color: #1E1E1E;
        }
        
        /*  Title  */
        .custom-title {
        font-size: 42px;
        font-family: 'Arial', sans-serif;
        word-wrap: break-word;
        white-space: normal;
        text-align: center;
        margin-bottom: 1rem;
        background-color: #121212;
        padding: 10px;
        font-weight: bold;
        text-decoration: underline;
        text-decoration-color: #FF8000;
        text-decoration-thickness: 0.15em;
    }

        </style>
        """, unsafe_allow_html=True
    )

    st.title("Jayant Chawla")
    st.markdown('<div class="custom-title">Optimizing car performance in Formula 1 through AI-Driven Telemetry Data Analysis</div>', unsafe_allow_html=True)
    
    

    selected = option_menu(
        menu_title="", 
        options=["Home", "Data Viewer", "Data Analysis"],
        icons=["house", "table", "graph-up"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",  
        styles={
            "nav-link": {
                "background-color": "#333333",  
                "color": "white",               
                "font-size": "16px",
                "margin": "0px 5px",
                "padding": "5px 10px",
            },
            "nav-link-selected": {
                "background-color": "#FF8000",  
                "color": "white",               
            },
            "container": {
                "padding": "5px",
                "background-color": "#121212",  
            },
            "icon": {
                "color": "white",
                "font-size": "18px",
            },
        }
    )
    
    if selected == "Home":
        st.subheader("Welcome to the Formula 1 Data Analysis App")
        st.write("The focus of this interactive app is to showcase the analysis of telemetry data from Formula 1 with the help of AI and ML")

        st.markdown("""
            - Python libraries used: Streamlit, Pandas, Numpy, FastF1, Matplotlib, Scikit-learn
            - Data Source: Fast-F1 API
            - Data Types: Telemetry, Weather, Tire
            - Analysis: Regression, Classification, Clustering, Anomaly Detection, Adaptive Strategies, Visualization
            - Github Repository: [F1-Telemetry-Analysis](https://github.com/JayantC31)
            - Author: Jayant Chawla
            """)

    elif selected == "Data Viewer":
        st.subheader("View raw Telemetry or External data") 
        data_choice = st.radio("What type of data do you want to view?",
                               ("Telemetry Data", "Weather Data", "Tire Data"))
        if data_choice == "Telemetry Data":
            telemetry_menu()
        elif data_choice == "Weather Data":
            external_menu("weather")
        elif data_choice == "Tire Data":
            external_menu("tire")
        else:
            st.error("Invalid selection")
    
    elif selected == "Data Analysis":
        st.subheader("View Analysis of Telemtry and External data") 
        analysis_option = st.radio(
            "Select the type of analysis you want to perform:",
            (
                "Select Option:",
                "Track Characteristics",
                "Pre-Race Strategy Optimization",
                "Comparative Analysis of Driver Performance Profiles"
                
                
            )
        )
           
                

        
    

        if analysis_option == "Select Option:":
            st.write("Please select an analysis option.")
            
        elif analysis_option == "Track Characteristics":
            st.divider()
            
            placeholder = st.empty()
            placeholder.write("Loading analysis for PCA and computing clustering method for circuit differentiation...")

            
            placeholder.empty()
            st.write("PCA analysis complete. See the visualization and insights below: ")
            st.write("")
            

            

            st.write("After basic data preprocessing, the data can be split into 3 main components(PC1,PC2,PC3) which explain most of the variance in the data.")
            st.write("The variance ratio can be shown as: ")

        
            html_table = """
            <table style="width:50%; border-collapse: collapse;" border="1">
                <tr>
                    <th>Value</th>
                    <th>Variance Captured</th>
                </tr>
                <tr>
                    <td style="text-align:center;">PC1</td>
                    <td style="text-align:center;">0.2802</td>
                </tr>
                <tr>
                    <td style="text-align:center;">PC2</td>
                    <td style="text-align:center;">0.2018</td>
                </tr>
                <tr>
                    <td style="text-align:center;">PC3</td>
                    <td style="text-align:center;">0.1205</td>
                </tr>
            </table>
            """

            # Display in Streamlit
            st.markdown(html_table, unsafe_allow_html=True)


            st.write("The PCA components based on each variable can be shown as: ")

            loadings = pd.read_csv("2025-04-21T20-44_export.csv")
            st.dataframe(loadings)
            st.write("        - For PC1, the engine performance and variable driving styles cause the most effect, with the circuits that are characterized by sustained high throttle usage, elevated RPM, and longer track lengths, are seen to have an increased PC1 level. However, it seems to have a negative loading on lower gear ratios, which means a low PC1 could mean the track is more brake-intensive \n This suggests that on these circuits, drivers adopt an aggressive driving style, and maintaining high power output — which can come at the cost of increased tyre wear and potentially greater engine stress, however the engine stress can not be seen from the telemetry data.")
            st.write("        - PC2 is mainly differenciated with the circuit’s layout. There are high positive loadings on the track's lap time, track length, and the straight line measures which include the max straight length and total straight distance. There are some negative contributions from a higher number of turns and low-speed corners which suggests that the lower the PC2 loading, the more technical the average time spent in the circuit is.")
            st.write("        - PC3 is very much influenced by atmospheric pressure, and the elevation, which means it takes into account the location of the circuits, with significant negative loading from the track's average elevation. Also, factors like the number of turns and higher gear ratios contribute to PC3. Thus, a high PC3 score points to circuits where environmental conditions and technical cornering demands are more pronounced, while lower PC3 scores reflect circuits with lower elevation variation and less technical complexity.")

            pcaimg = Image.open("Screenshot 2025-03-26 151118.png")

            st.image(pcaimg, use_container_width =True)

            st.write("")
            st.write("This 3d PCA plot provides a visual summary of the complex telemetry data gathered by the cars, and highlights the most important patterns and relationships among circuits.")
            st.write("With this analysis, we can make some assumptions based on each circuit's unique characteristics and location on this graph: ")
            st.write("For example, Monaco, Azerbaijan, Las Vegas and Mexico are clear outliers, and we can make some clear assumptions about their unique characteristics based on the PCA components.")
            st.write("        - Monaco is a very technical circuit with a lot of elevation changes, and a lot of low-speed corners, which is why it has such a low PC2 score and high PC3 score.")
            st.write("        - The Azerbaijan and Las Vegas tracks are very similar in which they are both street circuits with similar characteristics, which is why they are so close to each other in the PCA plot. They have a lot of high speed sections which requires higher gears and long straights, which is why they have a high PC2 score.")            
            st.write("        - The mexican circuit has a very high altitude and elevation compared to sea level, which is why it has a very low PC3 score, and it has a low PC1 score as it is a brake intensive circuit.")
            st.divider()
            
            
            st.write("The Elbow Method: ")
            #st.image(buf, width=600)
            
            st.write("The reason for the elbow method is to determine the optimal number of clusters for the clustering algorithm used in this analysis.")
            
            elbowimage = Image.open("Screenshot 2025-03-26 151435.png")

            st.image(elbowimage, use_container_width =True)


            st.divider()
            
            clusterimage = Image.open("Screenshot 2025-03-26 152742.png")

            st.image(clusterimage, use_container_width =True)

            st.write("This KMeans clustering algorithm showcases the different clusters of circuits based on the PCA components. The clusters are based on the similarity of the circuits in terms of the telemetry data gathered.")
            st.write("By reducing the high-dimensional telemetry data to its most significant components using PCA, we can effectively group circuits that share similar performance characteristics. ")
            st.write("These insights can help teams understand the underlying patterns in circuit performance, potentially guiding race strategies and car setups given each circuit's unique characteristics.")


        elif analysis_option == "Pre-Race Strategy Optimization":
            st.divider()
            st.write("Pre-Race strategy setup tool")
            if st.button("Reset Settings"):
                for key in ["settings"]:
                    if key in st.session_state:
                        del st.session_state[key]

            if "settings" not in st.session_state:
                settings = CustomOptions(require_session=False,require_lap=False)
                if settings is not None:
                    st.session_state.settings = settings
            else:
                settings = st.session_state.settings

            if settings:
                settingsdata = stg.getDataFromCsv(settings)
                st.divider()
                st.subheader("Violin Plots")
                selected_metrics = st.selectbox(
                    "Select metrics to visualize",
                    options=["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"],
                    key="selected_metrics"
                )
                
                st.write("The following violin plots showcase the distribution of the selected metrics across the drivers selected.")
                st.write("This helps to identify the differneces in the average and spread of the laptimes between the drivers distributed by the stints.")
                if selected_metrics:
                    fig = stg.violinplots(settingsdata, metrics=selected_metrics)
                    if fig is not None:
                        st.pyplot(fig)
                    else:
                        st.write("No stints with at least 2 drivers available for plotting.")
                
                st.divider()
                st.subheader("Session pace analysis")

                fig = stg.plot_session_pace_laptimes(settingsdata)  
                st.pyplot(fig)
                st.write("This session pace graph shows the average lap time of each driver over the course of the session.")
                st.write("The circles represent the outliers whereas the pitstop markers represent the pitstops made by the drivers.")
                st.write("This graph can be used to identify the drivers who are consistently fast or slow, and can help in making strategic decisions and also show "
                "the consistency of the drivers over the course of the session.")
                st.write("This also helps the team understand in which areas the driver are more likely losing time and at what points they are gaining time.")
                st.write("This also helps teams understand tyre degradation and the effect of the tyres on the lap times assuming they believe the drivers laptimes are consistent"
                "between laps, however this data is not publicaly available, which means the analysis of these graphs must be taken with some caution due to potential inaccuracies or unaccounted for variables.")
                
                st.divider()
                processlapdata = stg.prepare_driver_data_for_tyre_analysis(settingsdata)
                getweatherdata = stg.get_weather_data(processlapdata, settings)
                CombinedDFforStrat = stg.combine_weather_and_driver_data(processlapdata, getweatherdata)
                
                st.subheader("Regression Analysis")
                st.write("The following regression analysis showcases the relationship between selected features and the target variable which is the LapTime.")
                stratcombinedmetrics, figs = stg.regression_analysis_all_drivers_rf_with_features(CombinedDFforStrat, target="LapTime")

                st.write("Below are the evaluation metrics for predicting lap times using a Random Forest regression model for each driver. "
                "These metrics help assess the accuracy and reliability of the predictions:")
                st.write("- Mean Absolute Error (MAE): Average absolute deviation between predicted and actual lap times.")
                st.write("- Mean Squared Error (MSE): Average squared deviation between predicted and actual lap times.")
                st.write("- R² Score**: The proportion of variance that model can explain")
                st.write("- Out-of-Bag (OOB) Score: A measure of the model's accuracy in predicting unseen data.")
                

                table_data = []
                for driver, driver_metrics in stratcombinedmetrics.items():
                    stratcombinedmetrics_data = driver_metrics["RF"]
                    table_data.append({
                        "Driver": driver.replace('_', ' '),
                        "MAE (s)": f"{stratcombinedmetrics_data['MAE']:.3f}",
                        "MSE (s²)": f"{stratcombinedmetrics_data['MSE']:.3f}",
                        "R² Score": f"{stratcombinedmetrics_data['R2']:.3f}",
                        "OOB Score": f"{stratcombinedmetrics_data['OOB']:.3f}"
                    })

                df_metrics = pd.DataFrame(table_data)

                st.dataframe(df_metrics.style.set_properties(**{
                    'font-size': '20px',
                    'text-align': 'center'
                }), use_container_width=True)

                st.divider()

                for fig in figs:
                    st.pyplot(fig)
                
                st.write("The plot on the left shows the actual lap times vs the predicted lap times for each driver. "
                "The closer the points are to the line, the better the model is at predicting the lap times.")
                st.write("This uses a regression model to predict the lap times based on the selected features, and the model's accuracy is evaluated using the metrics shown above.")
                st.write("The plot on the right is a residual plot, which shows the difference between the actual and predicted lap times per driver. "
                         "This helps to identify patterns in the model's errors and can be used to improve the model's accuracy.")
                st.divider()
                st.subheader("Pitstop Strategy Recommendations")
                st.write("The following analysis showcases the optimal pitstop strategy for each driver based on the combined dataframe which includes the telemetry data and weather conditions.")
                pit_stop_recommendations = stg.find_optimal_pit_stop_tree(CombinedDFforStrat)
                for driver, recs in pit_stop_recommendations.items():
                    st.markdown(f"## 🏁 **Driver: {driver.replace('_', ' ')}**")
                    
                    if recs:
                        for rec in recs:
                            stint = int(rec['Stint'])
                            recommended_lap = int(rec['RecommendedPitLap'])
                            degradation = rec['PredictedDegradation']

                            st.markdown(f"""
                            **Stint:** `{stint}`  
                            - **Recommended Pit Lap:** `{recommended_lap}`  
                            - **Predicted Degradation:** `{degradation:.2f} seconds`
                            """)

                    else:
                        st.info("No pit stop recommendations available for this driver.")

                

        elif analysis_option == "Comparative Analysis of Driver Performance Profiles":
            st.divider()
            st.write("Driver Comparison Analysis Tool")
            
            if st.button("Reset Settings"):
                for key in ["analysis_settings", "driver_data", "aligned_data"]:
                    if key in st.session_state:
                        del st.session_state[key]
                
            
            if "analysis_settings" not in st.session_state:
                settings = CustomOptions(require_session=True, require_lap=True)
                if settings is not None:
                    st.session_state.analysis_settings = settings
            else:
                settings = st.session_state.analysis_settings
            
            if settings is not None:
                st.header("Comparative Analysis of Driver Performance Profiles")
                st.divider()

                st.subheader("Original Telemetry Graphs")
                st.write("The following analysis showcases the comparison of driver performance profiles based on telemetry data.")
                st.write("This includes comparing braking points, throttle usage, RPM levels, gear shift times, and speed profiles.")
                if "driver_data" not in st.session_state:
                    st.session_state.driver_data = dc.getDataFromCsv(settings)
                driver_data = st.session_state.driver_data
                if "aligned_data" not in st.session_state:
                    st.session_state.aligned_data = dc.align_telemetry_data(driver_data, num_points=1000)
                aligned_data = st.session_state.aligned_data
                feature_df = dc.extract_features(aligned_data)
                fig_original = dc.BeforeGRAPHS(driver_data)
                st.pyplot(fig_original)
                st.write("Based on the original telemetry data, we can see the differences in driving styles between the drivers.")
                
                st.divider()

                st.subheader("Time-Series Difference Plot")
                st.write("Pick an option to compare the time-series differences between the drivers, which can help identify key differences in their driving styles.")
                st.write("The data is aligned based on the distance covered during the lap.")
                metric_option = st.selectbox(
                    "Select a metric to compare:", 
                    options=["Throttle_x", "Speed_x", "Brake_x", "RPM_x"],
                    index=0,
                    key="ts_metric"
                )
                fig_diff = dc.plot_time_series_differences(aligned_data, metric=metric_option)
                if fig_diff:
                    st.pyplot(fig_diff)
                st.write("This plot is useful as it helps identify key differences during the lap")
                st.write("It emphasizes the rate of change and helps make it easier to spot sudden shifts in the data that could be"
                " indicative of a driver's or the cars performance.")
                st.divider()
                
                st.subheader("Radar Chart")
                st.write("This tool can be used to display multivariate data in a way that highlights the differences and similarities between the drivers telemetry "
                "across several variables at once. This can help identify the key areas where drivers differ in their driving styles.")
                dc.plot_radar_chart_scaled(feature_df)
                st.pyplot(plt.gcf())
                plt.clf()
                st.write("This chart helps show the differences that the drivers have in their driving styles during a lap, and can help identify key "
                "differences in their performance.")
                st.write("The chart stems from -2 at the center and extends to 2 at the outer edge. This scaling was chosen in order to normalize the data,"
                " so that each value represents a standardized deviation from the mean, allowing for an easy comparison across the features.")
                
                st.divider()

                # get number of drivers from settings
                # we need more then 2 drivers for the pca chart to work
                if len(settings["Drivers"]) > 2:
                    st.subheader("PCA Chart")
                    st.write("Using PCA to analyse the telemetry data helps to identify the key features that differentiate the drivers.")
                    st.write("By reducing multiple key features into two principal components, we can visualize the data in a lower-dimensional space.")
                    st.write("This helps to show in which ways the drivers differ during a lap, and can show if they can be grouped based"
                             " on their driving styles and the cars performance capabilities.")
                    fig_pca, variance_ratio, feature_table = dc.PcaAnalysis(aligned_data)
                    st.pyplot(fig_pca)
                    st.write("Using the variance, ratio, it is clear that from the two components that the first component explains the most variance in the data"
                             "but also that the variance can mostly be explained by the first two components.")
                    st.write("The variance ratio can be shown as:")
                    pcacol1,pcacol2 = st.columns(2)
                    with pcacol1:
                        st.write("PC1: " + f"{variance_ratio[0]:.3f}")
                    with pcacol2:
                        st.write("PC2: " + f"{variance_ratio[1]:.3f}")

                    st.write("The features that most correlate to each PCA component based on each variable can be shown as: ", feature_table)
                    top_features_df = dc.get_top_features_for_pcs(feature_table, top_n=5)
                    pc1_high = top_features_df[(top_features_df["PCA Component"] == "PC1") & (top_features_df["Direction"] == "High")]
                    pc1_low  = top_features_df[(top_features_df["PCA Component"] == "PC1") & (top_features_df["Direction"] == "Low")]
                    pc2_high = top_features_df[(top_features_df["PCA Component"] == "PC2") & (top_features_df["Direction"] == "High")]
                    pc2_low  = top_features_df[(top_features_df["PCA Component"] == "PC2") & (top_features_df["Direction"] == "Low")]
                    st.write("To further understand the PCA components, we can look at the top features that contribute to each component:")
                    st.write("The top 5 features that contribute to each PCA component are:")
                    pccol1, pccol2 = st.columns(2)
                    pccol3, pccol4 = st.columns(2)
                    with pccol1:
                        st.write("PC1 High")
                        html_table = pc1_high[["Feature", "Loading"]].to_html(index=False)
                        st.write(html_table, unsafe_allow_html=True)

                    with pccol2:
                        st.write("PC1 Low")
                        html_table = pc1_low[["Feature", "Loading"]].to_html(index=False)
                        st.write(html_table, unsafe_allow_html=True)

                    with pccol3:
                        st.write("PC2 High")
                        html_table = pc2_high[["Feature", "Loading"]].to_html(index=False)
                        st.write(html_table, unsafe_allow_html=True)
                    
                    with pccol4:
                        st.write("PC2 Low")
                        html_table = pc2_low[["Feature", "Loading"]].to_html(index=False)
                        st.write(html_table, unsafe_allow_html=True)
                else:
                    st.write("Please select at least 3 drivers to view the PCA chart as PCA components do not work for only 2 drivers")

        
            
            
        else:
            st.error("Invalid selection")

        
if __name__ == "__main__":
    main()

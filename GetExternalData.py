
import fastf1 as ff1
import pandas as pd
import os
import logging
import warnings
import time

# This is to supress the logs when running code in the terminal 
logging.getLogger("fastf1").setLevel(logging.ERROR)
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# enable the cache where meta data is collected assuming cache was made in main python file
ff1.Cache.enable_cache('cache')  

# make the folder to store the information
output_folder = "External_Data"
os.makedirs(output_folder, exist_ok=True)

# List of all possible session types using the fastf1 api names
SessionLists = ['Q', 'R','SQ', 'SPRINT']  

# in case i need to retyr
MAXRETRY = 5  
RETRYTIME = 10


def get_yearly_data(year,first_round_num,total_rounds):
    """Get the weather and tire data for a given year

    """
    # per each round, get the weather and tire data
    for round_number in range(first_round_num, total_rounds): 
        WeatherArray = []  
        TireArray = []  
        
        for Sessiontype in SessionLists:
            retries = 0 # start with 0 retries
            while retries < MAXRETRY:
                try:
                    # load the session data and ignore telemetry data
                    session = ff1.get_session(year, round_number, Sessiontype)
                    round_name = session.event['EventName']
                    session.load(telemetry=False, weather=True)
                    
                    # get the weather data
                    if session.WeatherDATA is not None:
                        WeatherDATA = session.WeatherDATA
                        weather_df = pd.DataFrame(WeatherDATA) # use pandas to make a dataframe
                        weather_df['Round'] = round_number
                        weather_df['Session'] = Sessiontype
                        WeatherArray.append(weather_df)

                    # do the same but for tires
                    if session.laps is not None:
                        # the specific data that i need to check are the stint and compounds data and the correlated lap and pittimes
                        tireDATA = session.laps[['Driver', 'Stint', 'Compound', 'TyreLife', 'LapNumber', 'PitInTime', 'PitOutTime']].copy()
                        tireDATA.loc[:, 'Round'] = round_number
                        tireDATA.loc[:, 'Session'] = Sessiontype
                        TireArray.append(tireDATA)

                    break  
                except ValueError as e:
                    print(f"ValueError found in the Round {round_number}, Session {Sessiontype}: {e}")
                    break
                except Exception as e:
                    retries = retries + 1
                    print(
                        f"Problem getting data in Round {round_number}, Session {Sessiontype}"
                        f"Starting retry {retries}/{MAXRETRY} in {RETRYTIME} seconds. Error: {e}"
                    )
                    time.sleep(RETRYTIME)
                    if retries == MAXRETRY:
                        print(f"No more retries for Round {round_number}, Session {Sessiontype}")
                        # skip as there is assumption of no data or a problem with server side

        if WeatherArray:
            weatherdf = pd.concat(WeatherArray, ignore_index=True)
            round_weather_csv = os.path.join(output_folder, f"{year}_{round_name}_Weather_Data.csv") # save as a csv
            weatherdf.to_csv(round_weather_csv, index=False) # i want index false
            print(f"Saved weather data for Round {round_number} to file {round_weather_csv}") # print location of csv
        
        if TireArray:
            tiredf = pd.concat(TireArray, ignore_index=True)
            round_tire_csv = os.path.join(output_folder, f"{year}_{round_name}_Tire_Data.csv")
            tiredf.to_csv(round_tire_csv, index=False)
            print(f"Saved tire data for Round {round_number} to file {round_tire_csv}")
    

def run():
    """
    Run the data processing for the seasons
    This checks for the years that are needed, and saves the data to an output folder
    """
    # ask them for the year and then was data they want
    try:
        startyear = int(input("What year do you want to start with: "))
        multipleyears = input("Do you want to get data for multiple years (yes/no): ").strip().lower()
        
        if multipleyears in ["yes", "true", "y", "1"]:
            endyear = int(input("What year do you want to end with (Do same year as start year if you only want 1 year): "))
            if endyear < startyear:
                print("The end year cannot be before the start year")
                return
            if endyear > 2024:
                print("2024 is the last year possible")
                return
            if endyear == startyear:
                endyear += 1
        elif multipleyears in ["no", "false", "n", "0"]:
            endyear = startyear + 1
        else:
            print("Invalid input. Enter 'yes' or 'no'.")
            return
        # main loop to get the data
        for i in range(startyear, endyear):
            print(f"Starting Season: {i}")
            year = i
            # check how many rounds are in the season
            season_schedule = ff1.get_event_schedule(year)
            total_rounds = len(season_schedule)

            # get the first round that is not the pre-season test
            first_round_num = None
            # for each row in the events make sure to not inclulde the pre season tests
            for _, row in season_schedule.iterrows():
                if "pre-season" not in row['EventName'].lower():
                    first_round_num = row['RoundNumber']
                    break
            get_yearly_data(year,first_round_num,total_rounds)
    except ValueError:
        print("Please enter a valid year as an integer.")


# main function to run the code
if __name__ == "__main__":
    run()

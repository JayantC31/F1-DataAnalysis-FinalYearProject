import fastf1 as ff1
import pandas as pd
import os
import time
import logging
import warnings

# get the logs but only give me the errors
logging.getLogger("fastf1").setLevel(logging.ERROR)
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
# make cache
basedirectorry = os.path.dirname(os.path.abspath(__file__))
cachedirectorry = os.path.join(basedirectorry, "cache")
# if cache is made then enable or make it
if os.path.exists(cachedirectorry) and os.listdir(cachedirectorry):
    ff1.Cache.enable_cache('cache')
else:
    os.makedirs(cachedirectorry, exist_ok=True)
# use file telemetry data
OutFolder = "Telemetry_Data"
os.makedirs(OutFolder, exist_ok=True)


# session types, dont use FP1,FP2,FP3

sessionLIST = [ 'Q', 'R', 'SQ', 'SPRINT']
# retries and delay
MAX_RETRIES = 3
RETRY_DELAY = 10

def GetTheSessions(event, season):
    """
    return the session names that are available for the gp

    """
    # get the event name
    race_name = event['EventName']
    sessions = []
    # sprint weekends dont have FP3, or FP2
    is_fp2_found = False
    for SessionNAME in sessionLIST:
        if SessionNAME == "FP3" and not is_fp2_found:
            #print("not sprint weekend")
            continue

        try:
            session = ff1.get_session(season, race_name, SessionNAME)
            session.load()
            sessions.append(SessionNAME)
            
                #print(f"FP2 session found for {SessionNAME} in {race_name}")
        except Exception as e:
            print(f"Session {SessionNAME} for {race_name} was not found: {e}")
    # return the sessions for each gp
    return sessions

def saveSessionData(session, SessionNAME, driver, DriverDIR):
    """
    Save laps and telemetry data for each driver and session
    
    
    """
    print(f"\nLoading: {SessionNAME}..")
    print(f"Finding laps for driver {driver} in {SessionNAME}..")
    # try to get the laps for the driver
    attempt = 0 
    while attempt < MAX_RETRIES:
        try:
            # get the laps for the driver
            laps_driver = session.laps.pick_driver(driver)
            break  
        except Exception as e:
            attempt += 1
            print(f"Laps data not loaded for driver {driver} in {SessionNAME} (attempt {attempt}/{MAX_RETRIES}): {e}")
            time.sleep(RETRY_DELAY)
    else:
        print(f"Not found, Skipping driver {driver} for session {SessionNAME} after {MAX_RETRIES} attempts")
        return
    # if the lap is deleted then delete the row
    if 'Deleted' in laps_driver.columns:
        laps_driver = laps_driver[laps_driver['Deleted'] != True]

    # save the lap data as laps csv
    lap_data_file = os.path.join(DriverDIR, f"{SessionNAME}_laps.csv")
    # remove these columns
    REMOVECOLS = ['DriverNumber', 'Team', 'TrackStatus', 'DeletedReason', 'FastF1Generated', 'IsAccurate']
    laps_driver_clean = laps_driver.drop(
        columns=[col for col in REMOVECOLS if col in laps_driver.columns],
        errors='ignore'
    )
    # save to the csvs
    laps_driver_clean.to_csv(lap_data_file, index=False)
    print(f"Lap data saved to: {lap_data_file}")
    # save the telemetry data
    telemetry_dir = os.path.join(DriverDIR, SessionNAME, "Telemetry")
    os.makedirs(telemetry_dir, exist_ok=True)
    for lap in laps_driver.iterlaps():
        lap_row = lap[1]
        # get the lap number as the key
        lap_number = lap_row['LapNumber']
        # if the lap is deleted then skip the telemetry
        if 'Deleted' in lap_row and str(lap_row['Deleted']).lower() == 'true':
            continue

        print(f"Getting telemetry for lap {lap_number} in {SessionNAME}")
        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                car_data = lap_row.get_car_data()
                positional_data = lap_row.get_telemetry()
                break
            except Exception as e:
                attempt += 1
                print(f"Telemetry data not loaded for lap {lap_number} in {SessionNAME} (attempt {attempt}/{MAX_RETRIES}): {e}")
                time.sleep(RETRY_DELAY)
        else:
            print(f"Telemetry not found, Skipping the lap {lap_number} in {SessionNAME} after {MAX_RETRIES} attempts")
            continue

        if car_data.empty or positional_data.empty:
            print(f"There is no telemetry data found for lap {lap_number} in {SessionNAME}")
            continue

        merged_telemetry = positional_data.merge(car_data, on="Time", how="inner")
        # remove these columns from the telemetry
        REMOVETelemetryCOLS = ['Date_y', 'RPM_y', 'Speed_y', 'nGear_y', 'Throttle_y', 'Brake_y', 
                                   'DRS_y', 'DriverAhead', 'DistanceToDriverAhead', 'Status']
        merged_telemetry_clean = merged_telemetry.drop(
            columns=[col for col in REMOVETelemetryCOLS if col in merged_telemetry.columns],
            errors='ignore'
        )
        # save to csv
        telemetry_file = os.path.join(telemetry_dir, f"Lap_{lap_number}_telemetry.csv")
        merged_telemetry_clean.to_csv(telemetry_file, index=False)
        print(f"Telemetry for lap {lap_number} saved to {telemetry_file}") 

def determine_start_round(season, schedule):
    """
    get the last complete round and start from there
    
    
    """
    # sort by the round number to get the last gp with full data
    schedule.sort_values("RoundNumber", inplace=True)
    events_in_season = []
    for _, event in schedule.iterrows():
        if "pre-season" in event['EventName'].lower():
            continue
        # skip pre season when starting
        round_number = event['RoundNumber']
        folder_name = event['EventName'].replace(" ", "_")
        events_in_season.append((round_number, folder_name))
    # get the drivers in the season
    DriverDIRs_all = [d for d in os.listdir(OutFolder) if os.path.isdir(os.path.join(OutFolder, d))]
    drivers_in_season = []
    event_folder_set = {folder for (_, folder) in events_in_season}
    # for each driver in the season check if the event folder is in the driver folder
    for d in DriverDIRs_all:
        driver_path = os.path.join(OutFolder, d)
        subdirs = [sub for sub in os.listdir(driver_path) if os.path.isdir(os.path.join(driver_path, sub))]
        if any(sub in event_folder_set for sub in subdirs):
            drivers_in_season.append(d)
            
    majority = (len(drivers_in_season) // 2) + 1 if drivers_in_season else 1

    last_complete_round = 0
    # get the last complete round number and start from there
    for round_number, folder_name in events_in_season:
        count = 0
        for d in drivers_in_season:
            driver_event_path = os.path.join(OutFolder, d, folder_name)
            if os.path.exists(driver_event_path) and os.path.isdir(driver_event_path):
                if os.listdir(driver_event_path):
                    count += 1
        if count >= majority:
            last_complete_round = round_number
        else:
            break


    # start from the last complete round
    start_round = last_complete_round 
    if last_complete_round == 0:
        print("no data found for the season")
    print(f"Detected last complete round: {last_complete_round}. Starting from round: {start_round}")
    return start_round

def process_season(season):
    """Process and save data for a given season"""
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            schedule = ff1.get_event_schedule(season)
            # get the schedule
            break
        except Exception as e:
            attempt += 1
            print(f"Failed to retrieve event schedule for season {season} (attempt {attempt}/{MAX_RETRIES}): {e}")
            time.sleep(RETRY_DELAY)
    else:
        print(f"Not found, Skipping season {season} after {MAX_RETRIES} attempts to retrieve event schedule")
        return
    # strat from the last complete round
    start_round = determine_start_round(season, schedule)
    
    for _, event in schedule.iterrows():
        if "pre-season" in event['EventName'].lower():
            continue
        # skip the events that are already processed    
        if event['RoundNumber'] < start_round:
            print(f"Skipping {event['EventName']} (round {event['RoundNumber']}) as data already exists")
            continue

        race_name = event['EventName']
        sessions = GetTheSessions(event, season)
        print(f"\nProcessing {race_name} ({season})")
        for SessionNAME in sessions:
            try:
                # get the session given the season, round number and session name
                session = ff1.get_session(season, event['RoundNumber'], SessionNAME)
            except Exception as e:
                print(f"Failed to get session {SessionNAME} for {race_name}: {e}")
                continue

            attempt = 0
            loaded = False
            while attempt < MAX_RETRIES:
                try:
                    session.load()
                    loaded = True
                    break
                except Exception as e:
                    attempt += 1
                    print(f"Failed to load session {SessionNAME} for {race_name} (attempt {attempt}/{MAX_RETRIES}): {e}")
                    time.sleep(RETRY_DELAY)
            if not loaded:
                print(f"Skipping session {SessionNAME} for {race_name} after {MAX_RETRIES} attempts")
                continue

            attempt = 0
            while attempt < MAX_RETRIES:
                try:
                    # get the drivers in the session
                    drivers = session.laps['Driver'].unique()
                    break
                except Exception as e:
                    attempt += 1
                    print(f"Laps data not loaded for session {SessionNAME} in {race_name} (attempt {attempt}/{MAX_RETRIES}): {e}")
                    time.sleep(RETRY_DELAY)
            else:
                print(f"Skipping session {SessionNAME} for {race_name} after {MAX_RETRIES} attempts to access laps data")
                continue

            for driver in drivers:
                try:
                    # get the driver name and save to the file directory
                    driver_name = session.get_driver(driver)['FullName']
                except KeyError:
                    print(f"Failed to retrieve driver name for ID: {driver}")
                    continue

                DriverDIR = os.path.join(
                    OutFolder,
                    str(season),
                    driver_name.replace(" ", "_"),
                    race_name.replace(" ", "_")
                )
                os.makedirs(DriverDIR, exist_ok=True)
                saveSessionData(session, SessionNAME, driver, DriverDIR)

            del session

def run():
    """
    Run the data processing for the specified season(s).
    """
    # ask user how many years they want to get data for
    try:
        startyear = int(input("What year do you want to start with: "))
        multipleyears = input("Do you want to get data for multiple years (yes/no): ").strip().lower()
        if multipleyears in ["yes", "true", "y", "1"]:
            endyear = int(input("What year do you want to end with (enter the same year if you only want 1 year): "))
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
            print("Invalid input. Enter 'yes' or 'no'")
            return
        for year in range(startyear, endyear):
            print(f"Starting Season: {year}")
            process_season(year)
    except ValueError:
        print("Please enter a valid year as an integer")

if __name__ == "__main__":
    run()

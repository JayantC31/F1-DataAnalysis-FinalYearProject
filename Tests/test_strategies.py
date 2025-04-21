
import os
import tempfile
import unittest
import pandas as pd

from strategies import getDataFromCsv,prepare_driver_data_for_tyre_analysis,get_weather_data,combine_weather_and_driver_data, find_optimal_pit_stop_tree
# run with: python -m unittest Tests/test_strategies.py

class TestStrategies(unittest.TestCase):

    def test_getDataFromCsv(self):
        with tempfile.TemporaryDirectory() as tempp:
            cwd = os.getcwd()
            try:
                os.chdir(tempp)
                year = "2024"
                driver1 = "Driver_One"
                driver2 = "Driver_Two"
                race_event = "EventA"
                # create the csv needed 
                path = os.path.join("Telemetry_Data", year, driver1, race_event)
                os.makedirs(path, exist_ok=True)
                df = pd.DataFrame({"col": [1, 2]})
                df.to_csv(os.path.join(path, "R_laps.csv"), index=False)

                settings = {
                    "Year": year,
                    "Race Event": race_event,
                    "Drivers": [driver1, driver2]
                }
                result = getDataFromCsv(settings)

                self.assertIn(driver1, result)
                pd.testing.assert_frame_equal(result[driver1], df)
                self.assertNotIn(driver2, result)
            finally:
                os.chdir(cwd)

    def test_preprocessing_outliers(self):
        # check for outlierrs
        driver = "D1"
        df = pd.DataFrame({
            "LapNumber": [1, 2, 3],
            "LapTime": ["00:01:00", "00:01:10", "00:10:00"],
            "Stint": [1, 1, 1],
            "Compound": ["Soft", "Soft", "Soft"]
        })
        processed = prepare_driver_data_for_tyre_analysis({driver: df})
        self.assertIn(driver, processed)
        out_df = processed[driver]
        # outlier rap removed
        self.assertTrue((out_df["LapNumber"] <= 2).all())
        # check for fuel
        self.assertIn("FuelPercent", out_df.columns)
        self.assertEqual(len(out_df), 2)
        self.assertGreater(out_df["FuelPercent"].iloc[0],
                           out_df["FuelPercent"].iloc[1])

    def test_weather(self):
        with tempfile.TemporaryDirectory() as tempp:
            cwd = os.getcwd()
            try:
                os.chdir(tempp)
                os.makedirs("External_Data", exist_ok=True)
                year = "2024"
                race_event = "Test_Event"
                # Create CSV with three time points
                weather_df = pd.DataFrame({
                    "Time": ["00:00:05", "00:00:15", "00:00:25"],
                    "Session": ["R", "R", "R"],
                    "Temp": [100, 200, 300]
                })
                fname = f"{year}_{race_event.replace('_',' ')}_Weather_Data.csv"
                weather_df.to_csv(os.path.join("External_Data", fname),
                                  index=False)

                # Driver data spans 10s to 20s
                drv_df = pd.DataFrame({"Time": ["00:00:10", "00:00:20"]})
                processed = {"drv": drv_df}
                settings = {"Year": year, "Race Event": race_event}

                out = get_weather_data(processed, settings)
                # Only the 15s row falls between 10s and 20s
                self.assertTrue((out["Temp"] == 200).all())

                # Missing keys should raise
                with self.assertRaises(ValueError):
                    get_weather_data(processed, {})
            finally:
                os.chdir(cwd)

    def test_mergeddata(self):
        driver = "D1"
        df = pd.DataFrame({
            "Time": ["00:00:10", "00:00:20"],
            "LapNumber": [1, 2]
        })
        weather = pd.DataFrame({
            "Time": ["00:00:15"],
            "Rain": [5]
        })
        merged = combine_weather_and_driver_data({driver: df}, weather)
        self.assertIn(driver, merged)
        mdf = merged[driver]
        self.assertIn("Rain", mdf.columns)
        self.assertEqual(len(mdf), 2)
    # check in future for the optimal lap time for the strategies per race different per season and tyre compound
    def test_optimalpitstop(self):
        driver = "D1"
        # 1 lap
        df = pd.DataFrame({
            "LapNumber": [1],
            "Stint": [1],
            "LapTime": ["00:01:00"]
        })
        recs = find_optimal_pit_stop_tree({driver: df}, window_size=2)
        self.assertIn(driver, recs)
        self.assertEqual(recs[driver], [])

        # check for missing
        MISSINGreacs = find_optimal_pit_stop_tree({"X": pd.DataFrame({"A": [1]})})
        self.assertNotIn("X", MISSINGreacs)

if __name__ == "__main__":
    unittest.main()

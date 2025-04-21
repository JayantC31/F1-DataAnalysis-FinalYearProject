# run with: python -m unittest Tests/test_drivercomparisons.py
import os
import tempfile
import unittest
import numpy as np
import pandas as pd
import drivercomparisons as dc 

class TestDriverComparisons(unittest.TestCase):

    def test_getdata(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                # prepare csv
                base = os.path.join("Telemetry_Data", "2024", "DriverA", "Race1", "R", "Telemetry")
                os.makedirs(base, exist_ok=True)
                df = pd.DataFrame({"X": [1,2,3]})
                csv_path = os.path.join(base, "Lap_1.0_telemetry.csv")
                df.to_csv(csv_path, index=False)

                settings = {
                    "Year": "2024",
                    "Race Event": "Race1",
                    "Drivers": ["DriverA", "DriverB"],
                    "Session Summary File": "R_summary.csv",
                    "Lap Number": 1
                }
                result = dc.getDataFromCsv(settings)
                #  non-empty dataframe
                self.assertIn("DriverA", result)
                pd.testing.assert_frame_equal(result["DriverA"]["telemetry"], df)
                # empty 
                self.assertIn("DriverB", result)
                self.assertTrue(result["DriverB"]["telemetry"].empty)
            finally:
                os.chdir(cwd)

    def test_align(self):
        # driver1: 0,10,20; driver2: 5,15,25
        df1 = pd.DataFrame({"Distance": [0,10,20], "Speed_x": [0,1,2]})
        df2 = pd.DataFrame({"Distance": [5,15,25], "Speed_x": [5,6,7]})
        aligned = dc.align_telemetry_data({
            "D1": {"telemetry": df1},
            "D2": {"telemetry": df2}
        }, num_points=10)
        self.assertIn("D1", aligned)
        self.assertIn("D2", aligned)
        # each telemetry now has 10 rows
        self.assertEqual(len(aligned["D1"]["telemetry"]), 10)
        

    def test_features(self):
        df = pd.DataFrame({
            "Distance":[0,1,2,3],
            "nGear_x":[1,2,2,1],
            "Brake_x":[0,1,0,1],
            "Throttle_x":[0.5,1.0,0.0,0.8],
            "Speed_x":[100,110,105,115],
            "RPM_x":[5000,5500,5300,6000],
        })
        feats = dc.extract_features({"D1": {"telemetry": df}})
        # must have one row, index 'D1'
        self.assertListEqual(list(feats.index), ["D1"])
        # check some expected columns
        expected = {
            "avg_gear", "gear_shift_freq", 
            "brake_usage_mean", "brake_freq",
            "avg_throttle", "frac_full_throttle",
            "avg_speed", "max_speed",
            "avg_acceleration", "avg_deceleration",
            "avg_rpm", "rpm_ramp_up_rate", "rpm_ramp_down_rate"
        }
        self.assertTrue(expected.issubset(set(feats.columns)))

    def test_scaling(self):
        # fake features
        df = pd.DataFrame({
            "Distance":[0,1,2],
            "nGear_x":[1,2,1],
            "Brake_x":[0,1,0],
            "Throttle_x":[0.2,0.8,0.5],
            "Speed_x":[90,95,100],
            "RPM_x":[4000,4500,4200],
        })
        driver_data = {"D1": {"telemetry": df}, "D2": {"telemetry": df}}
        fig, var_ratio, feat_table = dc.PcaAnalysis(driver_data)
        # variance_ratio length 2
        self.assertEqual(len(var_ratio), 2)
        self.assertListEqual(list(feat_table.columns), ["PC1","PC2"])
        # sds check for driverdata
        feats = dc.extract_features(driver_data)
        self.assertTrue(set(feats.columns), set(feat_table.index))

   
    
if __name__ == "__main__":
    unittest.main()

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch

# need all these imports
from TrackStats import (
    count_turns,
    get_straight_metrics,
    compute_gear_distribution,
    aggregate_weather_features,
    aggregate_telemetry_features,
    processRaceData)


# run using python -m unittest Tests/test_TrackStats.py
class TestTrackStats(unittest.TestCase):

    def test_turns(self):
        # east and north for 1 turn
        df = pd.DataFrame({
            'X': [0, 1, 2, 2, 2],
            'Y': [0, 0, 0, 1, 2]
        }, index=[0,1,2,3,4])
        turns = count_turns(df, angle_threshold=10)
        self.assertEqual(turns, 1)

    def test_straights(self):
        # make a straight using coords
        datadata = pd.DataFrame({
            'X': [0, 1, 2, 2, 2],
            'Y': [0, 0, 0, 1, 2]
        }, index=[0,1,2,3,4])
        max_s, total_s = get_straight_metrics(datadata, angle_threshold=5)
        self.assertEqual(max_s, 2.0)
        self.assertEqual(total_s, 3.0)

    def test_gears(self):
        # random samples for gears to make distribution
        datadata = pd.DataFrame({'nGear_x': [1,2,2,3,3,3]})
        dist = compute_gear_distribution(datadata, gear_range=range(1,5))
        # ratios
        self.assertEqual(dist['gear1_ratio'], 1/6)
        self.assertEqual(dist['gear2_ratio'], 2/6)
        self.assertEqual(dist['gear3_ratio'], 3/6)
        self.assertEqual(dist['gear4_ratio'], 0.0)
        assert dist['gear4_ratio'] == 0.0  # nothing in 4th gear


    def test_weather(self):
        data = pd.DataFrame({
            'race': ['A','A','B','B'],
            'Humidity': [10,20,30,50],
            'Pressure': [1000,1020,990,1010]
        })
        agg = aggregate_weather_features(data)
        # should have two rows, mean for A and B
        self.assertEqual(set(agg['race']), {'A','B'})
        rowA = agg[agg['race']=='A'].iloc[0]
        self.assertEqual(rowA['Humidity'], 15)
        self.assertEqual(rowA['Pressure'], 1010)

    def test_aggreagateddata(self):
        # create telemetry_df with two races, each with one lap row
        telemetry_df = pd.DataFrame({
            'race': ['R1','R1','R2'],
            'lap': [1,1,1],
            'Brake_x': [1,1,0],
            'Throttle_x': [0.5,0.5,0.2],
            'Distance': [100,100,80],
            'X': [0,1,0],
            'Y': [0,1,0],
            'Z': [5,5,3],
            'RPM_x': [1000,1200,1100],
            'nGear_x': [1,1,2]
        })
        agg = aggregate_telemetry_features(telemetry_df)
        # should have two rows for R1 and R2
        self.assertEqual(set(agg['race']), {'R1','R2'})
        self.assertIn('avg_num_of_brake_events', agg.columns)
        self.assertIn('avg_throttle_usage', agg.columns)
        self.assertIn('avg_turns', agg.columns)

    @patch('TrackStats.preprocessTelemetryData')
    @patch('TrackStats.aggregate_telemetry_features')
    @patch('TrackStats.aggregate_weather_features')
    @patch('TrackStats.StandardScaler.fit_transform', lambda self, X: X)
    def test_racedatapreprocessing(self, mock_agg_weather, mock_agg_telemetry, mock_pre):
        import pandas as pd
        cf = pd.DataFrame({'race': ['R1'], 'f1': [0.1]})
        cleaned_summary = cleaned_telemetry = pd.DataFrame()
        weather_df = pd.DataFrame()
        mock_pre.return_value = (cf, cleaned_summary, cleaned_telemetry, weather_df)
        # aggregated telemetry
        at = pd.DataFrame({'race': ['R1'], 't1': [0.2]})
        mock_agg_telemetry.return_value = at
        # aggregated weather
        aw = pd.DataFrame({'race': ['R1'], 'Humidity': [50], 'Pressure': [1000]})
        mock_agg_weather.return_value = aw

        final = processRaceData()
        # merged should contain columns from all
        self.assertIn('race', final.columns)
        self.assertIn('f1', final.columns)
        self.assertIn('t1', final.columns)
        self.assertIn('Humidity', final.columns)
        self.assertIn('Pressure', final.columns)
        # one row
        self.assertEqual(len(final), 1)



if __name__ == '__main__':
    unittest.main()

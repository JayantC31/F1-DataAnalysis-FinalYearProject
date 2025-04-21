# run by doing python -m unittest Tests/test_GetExternalData.py
import sys
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from GetExternalData import get_yearly_data

class TestGetExternalData(unittest.TestCase):

    @patch('GetExternalData.ff1.get_session')
    def test_get_yearly_data_good(self, mock_get_session):
        # temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            
            import GetExternalData
            GetExternalData.output_folder = tmpdir
            # fake event
            session = MagicMock()
            session.event = {'EventName': 'Test Grand Prix'}
            session.load.return_value = None

            session.WeatherDATA = [{'temp': 20}, {'temp': 21}]
            # laps is a DataFrame with the required columns
            session.laps = pd.DataFrame({
                'Driver': ['A', 'B'],
                'Stint': [1, 1],
                'Compound': ['Soft', 'Medium'],
                'TyreLife': [5, 6],
                'LapNumber': [10, 11],
                'PitInTime': [100.0, 200.0],
                'PitOutTime': [110.0, 210.0],
            })

            mock_get_session.return_value = session

            year = 2024
            get_yearly_data(year, first_round_num=1, total_rounds=2)

            files = os.listdir(tmpdir)
            weather_files = [f for f in files if f.endswith('_Weather_Data.csv')]
            tire_files = [f for f in files if f.endswith('_Tire_Data.csv')]

            self.assertEqual(len(weather_files), 1)
            self.assertEqual(len(tire_files), 1)

            weather_df = pd.read_csv(os.path.join(tmpdir, weather_files[0]))
            self.assertIn('temp', weather_df.columns)
            self.assertIn('Round', weather_df.columns)
            self.assertIn('Session', weather_df.columns)

            tire_df = pd.read_csv(os.path.join(tmpdir, tire_files[0]))
            for col in ['Driver', 'Stint', 'Compound', 'LapNumber', 'PitInTime', 'PitOutTime', 'Round', 'Session']:
                self.assertIn(col, tire_df.columns)

    # also neede to check for when the driver data is empty or when there is no data for the code to check and add using the api

    @patch('GetExternalData.ff1.get_session')
    def test_withnodata(self, mock_get_session):
        with tempfile.TemporaryDirectory() as tmpdir:
            import GetExternalData
            GetExternalData.output_folder = tmpdir

            session = MagicMock()
            session.event = {'EventName': 'No Data GP'}
            session.load.return_value = None
            session.WeatherDATA = None
            session.laps = None  # no lap data

            mock_get_session.return_value = session

            # run for one round
            get_yearly_data(2024, first_round_num=1, total_rounds=2)

            self.assertEqual(os.listdir(tmpdir), [])

if __name__ == '__main__':
    unittest.main()

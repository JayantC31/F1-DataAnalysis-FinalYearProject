import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# run by doing python Tests/test_GetData.py
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from GetData import GetTheSessions, determine_start_round


# make a fake class in order to check if the data can be used
class TestGetData(unittest.TestCase):

    @patch('GetData.ff1.get_session')
    def test_good_session(self, mock_get_session):
        mock_session = MagicMock()
        mock_session.load.return_value = True
        mock_get_session.return_value = mock_session

        event = {'EventName': 'Austrian Grand Prix'}
        season = 2024

        sessions = GetTheSessions(event, season)

        self.assertIn('Q', sessions)
        self.assertIn('R', sessions)

    @patch('GetData.ff1.get_session')
    def test_bad_session(self, mock_get_session):
        mock_get_session.side_effect = Exception("Session not found")

        event = {'EventName': 'Invalid Grand Prix'}
        season = 2024

        sessions = GetTheSessions(event, season)

        self.assertEqual(sessions, [])

    @patch('GetData.os.path.exists', return_value=True)
    @patch('GetData.os.path.isdir', return_value=True)
    @patch('GetData.os.listdir')
    def test_existing_data(self, mock_listdir, mock_isdir, mock_exists):
        # Simulated schedule for the 2024 season
        schedule = pd.DataFrame({
            'RoundNumber': [1, 2],
            'EventName': ['Bahrain GP', 'Saudi Arabian GP']
        })

        def testraces(path):
            if path.endswith("Telemetry_Data"):
                return ["Driver1", "Driver2"]
            elif "Driver1" in path and not path.endswith("Bahrain_GP"):
                return ["Bahrain_GP"]
            elif "Driver2" in path and not path.endswith("Bahrain_GP"):
                return ["Bahrain_GP"]
            elif path.endswith("Bahrain_GP"):
                return ["somefile.csv"]
            return []

        mock_listdir.side_effect = testraces

        start_round = determine_start_round(2024, schedule)

        self.assertEqual(start_round, 2)


if __name__ == '__main__':
    unittest.main()

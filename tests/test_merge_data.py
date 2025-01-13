import unittest
from unittest.mock import patch
import pandas as pd

import os
import sys

# Add the root project directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.merge_cleaned_data import DataMerger

class TestDataMerger(unittest.TestCase):

    @patch('pandas.read_csv')
    def test_load_data(self, mock_read_csv):
        # Mock data for train_data and store_data
        mock_train_data = pd.DataFrame({
            'Store': [1, 2],
            'Sales': [1000, 1500],
            'Customers': [200, 300]
        })
        mock_store_data = pd.DataFrame({
            'Store': [1, 2],
            'CompetitionDistance': [500.0, 700.0]
        })
        
        # Mocking the read_csv function to return mocked data
        mock_read_csv.side_effect = [mock_train_data, mock_store_data]
        
        # Create an instance of DataMerger
        merger = DataMerger(base_dir=".")
        merger.load_data()
        
        # Check if the data is loaded correctly
        pd.testing.assert_frame_equal(merger.train_data, mock_train_data)
        pd.testing.assert_frame_equal(merger.store_data, mock_store_data)

    def test_merge_data(self):
        # Mock data for train_data and store_data
        mock_train_data = pd.DataFrame({
            'Store': [1, 2],
            'Sales': [1000, 1500],
            'Customers': [200, 300]
        })
        mock_store_data = pd.DataFrame({
            'Store': [1, 2],
            'CompetitionDistance': [500.0, 700.0]
        })
        
        # Create an instance of DataMerger and set the mock data
        merger = DataMerger(base_dir=".")
        merger.train_data = mock_train_data
        merger.store_data = mock_store_data
        
        # Merge the data
        merged_data = merger.merge_data()
        
        # Expected merged result
        expected_merged_data = pd.DataFrame({
            'Store': [1, 2],
            'Sales': [1000, 1500],
            'Customers': [200, 300],
            'CompetitionDistance': [500.0, 700.0]
        })
        
        # Check if the merged data is correct
        pd.testing.assert_frame_equal(merged_data, expected_merged_data)

    @patch('pandas.DataFrame.to_csv')
    def test_save_data(self, mock_to_csv):
        # Mock merged data
        mock_merged_data = pd.DataFrame({
            'Store': [1],
            'Sales': [1000],
            'CompetitionDistance': [500.0]
        })
        
        # Create an instance of DataMerger
        merger = DataMerger(base_dir=".")
        
        # Call the save_data method
        merger.save_data(mock_merged_data)
        
        # Check that to_csv was called
        mock_to_csv.assert_called_once_with(merger.output_file, index=False)

    @patch('os.makedirs')
    @patch('pandas.read_csv')
    def test_load_and_merge_data(self, mock_read_csv, mock_makedirs):
        # Mock data
        mock_train_data = pd.DataFrame({
            'Store': [1, 2],
            'Sales': [1000, 1500]
        })
        mock_store_data = pd.DataFrame({
            'Store': [1, 2],
            'CompetitionDistance': [500.0, 700.0]
        })
        mock_read_csv.side_effect = [mock_train_data, mock_store_data]
        
        # Create an instance of DataMerger
        merger = DataMerger(base_dir=".")
        
        # Load data
        merger.load_data()
        
        # Merge data
        merged_data = merger.merge_data()
        
        # Save data
        merger.save_data(merged_data)
        
        # Check if os.makedirs was called (to ensure folder structure)
        mock_makedirs.assert_called_once()
        
        # Check if data is saved correctly by verifying to_csv call
        self.assertEqual(mock_read_csv.call_count, 2)

if __name__ == '__main__':
    unittest.main()


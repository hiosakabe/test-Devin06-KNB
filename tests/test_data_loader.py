import os
import pytest
import pandas as pd
from unittest.mock import patch, mock_open, MagicMock
from src.data_loader import load_race_data, preprocess_data


class TestDataLoader:
    """Test class for data_loader module"""

    @patch('pandas.read_csv')
    @patch('pandas.concat')
    def test_load_race_data_normal(self, mock_concat, mock_read_csv, sample_df):
        """Test normal case: all CSV files are loaded and concatenated"""
        # Setup mock returns
        mock_read_csv.return_value = sample_df.iloc[:1].copy()  # Return a single row for each file
        mock_concat.return_value = sample_df  # Return the full sample_df for concat result
        
        # Call the function
        result = load_race_data(data_dir="/fake/path")
        
        # Verify read_csv was called 5 times (once for each file)
        assert mock_read_csv.call_count == 5
        
        # Verify concat was called once with a list of 5 dataframes
        args, _ = mock_concat.call_args
        assert len(args[0]) == 5
        
        # Verify the result has the expected columns
        assert list(result.columns) == list(sample_df.columns)
        
        # Verify the result has the expected shape
        assert result.shape == sample_df.shape

    @patch('pandas.read_csv')
    def test_load_race_data_file_not_found(self, mock_read_csv):
        """Test error case: file not found exception is handled"""
        # Setup mock to raise FileNotFoundError on the third file
        def side_effect(file_path, **kwargs):
            if '2000-2005_race_result.csv' in file_path:
                raise FileNotFoundError(f"File not found: {file_path}")
            return pd.DataFrame({'dummy': [1]})
        
        mock_read_csv.side_effect = side_effect
        
        # Call the function and check that it raises the expected exception
        with pytest.raises(FileNotFoundError):
            load_race_data(data_dir="/fake/path")

    @patch('pandas.read_csv')
    def test_load_race_data_corrupted_file(self, mock_read_csv):
        """Test error case: corrupted CSV file exception is handled"""
        # Setup mock to raise pd.errors.ParserError on the second file
        def side_effect(file_path, **kwargs):
            if '1993-1999_race_result.csv' in file_path:
                raise pd.errors.ParserError(f"Error parsing file: {file_path}")
            return pd.DataFrame({'dummy': [1]})
        
        mock_read_csv.side_effect = side_effect
        
        # Call the function and check that it raises the expected exception
        with pytest.raises(pd.errors.ParserError):
            load_race_data(data_dir="/fake/path")

    @patch('os.path.dirname')
    @patch('os.path.abspath')
    def test_load_race_data_default_path(self, mock_abspath, mock_dirname, sample_df):
        """Test that default path is constructed correctly when data_dir is None"""
        # Setup mocks
        mock_abspath.return_value = "/fake/path/src/data_loader.py"
        mock_dirname.side_effect = ["/fake/path/src", "/fake/path"]
        
        with patch('pandas.read_csv', return_value=sample_df.iloc[:1].copy()):
            with patch('pandas.concat', return_value=sample_df):
                # Call the function with default data_dir
                result = load_race_data()
                
                # Verify the path construction
                mock_abspath.assert_called_once()
                assert mock_dirname.call_count == 2
                
                # Verify read_csv was called with the correct paths
                calls = pd.read_csv.call_args_list
                for call in calls:
                    args, _ = call
                    assert args[0].startswith("/fake/path/data/")

    def test_preprocess_data(self, sample_df):
        """Test data preprocessing function"""
        # Create a test dataframe with some null values
        test_df = sample_df.copy()
        test_df.loc[0, "Final Position"] = None  # Add a null in target column
        test_df.loc[1, "Horse Name"] = None  # Add a null in a string column
        
        # Call the function
        result = preprocess_data(test_df)
        
        # Verify rows with null target are removed
        assert len(result) < len(test_df)
        assert "Final Position" in result.columns
        
        # Verify all columns are numeric (after LabelEncoder)
        for col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col])
        
        # Verify no nulls remain
        assert not result.isnull().any().any()

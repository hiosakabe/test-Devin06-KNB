import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.feature_engineering import create_numeric_features, generate_features, FEATURE_COLUMNS


class TestFeatureEngineering:
    """Test class for feature_engineering module"""

    def test_create_numeric_features_returns_only_feature_columns(self, sample_df):
        """Test that create_numeric_features returns only columns in FEATURE_COLUMNS"""
        # Call the function
        result = create_numeric_features(sample_df)
        
        # Verify result contains only FEATURE_COLUMNS
        assert set(result.columns) == set(FEATURE_COLUMNS)
        
        # Verify no columns are missing
        assert len(result.columns) == len(FEATURE_COLUMNS)
        
        # Verify row count matches input
        assert len(result) == len(sample_df)

    def test_create_numeric_features_empty_dataframe(self):
        """Test that create_numeric_features handles empty DataFrames correctly"""
        # Create an empty DataFrame with all required columns
        empty_df = pd.DataFrame(columns=FEATURE_COLUMNS)
        
        # Call the function
        result = create_numeric_features(empty_df)
        
        # Verify result is also empty but has correct columns
        assert len(result) == 0
        assert set(result.columns) == set(FEATURE_COLUMNS)

    @patch('src.feature_engineering.tqdm')
    @patch('src.feature_engineering.Timer')
    def test_generate_features_maintains_row_count(self, mock_timer, mock_tqdm, sample_df):
        """Test that generate_features returns DataFrame with same row count"""
        # Setup mocks
        mock_tqdm.return_value = [create_numeric_features]
        mock_timer_instance = MagicMock()
        mock_timer.return_value.__enter__.return_value = mock_timer_instance
        
        # Call the function
        result = generate_features(sample_df)
        
        # Verify row count matches input
        assert len(result) == len(sample_df)
        
        # Verify all feature columns are present
        for col in FEATURE_COLUMNS:
            assert col in result.columns

    @patch('src.feature_engineering.tqdm')
    @patch('src.feature_engineering.Timer')
    def test_generate_features_empty_dataframe(self, mock_timer, mock_tqdm):
        """Test that generate_features handles empty DataFrames correctly"""
        # Create an empty DataFrame with all required columns
        empty_df = pd.DataFrame(columns=FEATURE_COLUMNS)
        
        # Setup mocks
        mock_tqdm.return_value = [create_numeric_features]
        mock_timer_instance = MagicMock()
        mock_timer.return_value.__enter__.return_value = mock_timer_instance
        
        # Call the function
        result = generate_features(empty_df)
        
        # Verify result is also empty but has correct columns
        assert len(result) == 0
        assert all(col in result.columns for col in FEATURE_COLUMNS)

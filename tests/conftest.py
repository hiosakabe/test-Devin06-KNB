import pytest
import pandas as pd


@pytest.fixture
def sample_df():
    """
    A fixture that returns a sample DataFrame with racing data.
    
    This fixture creates a small DataFrame with 2 rows containing the main columns
    used in racing data analysis. It's intended to be used across multiple test files.
    
    Returns:
        pandas.DataFrame: A sample DataFrame with racing data
    """
    data = {
        "Race PP ID": ["R20210101-01", "R20210101-02"],
        "Race ID": ["20210101-01", "20210101-02"],
        "Race Day": ["2021-01-01", "2021-01-01"],
        "Racecourse Name": ["Tokyo", "Osaka"],
        "Final Position": [1, 2],
        "Driver ID": ["D001", "D002"],
        "Starting Position": [3, 1],
        "Finishing Time": ["1:23.456", "1:24.789"],
        "Points": [25, 18],
        "Status": ["Finished", "Finished"],
        "Laps Completed": [60, 60],
        "Best Lap": ["1:22.345", "1:23.456"]
    }
    
    return pd.DataFrame(data)

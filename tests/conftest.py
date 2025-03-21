import pytest
import pandas as pd


@pytest.fixture
def sample_df():
    """
    A fixture that returns a sample DataFrame with comprehensive racing data.
    
    This fixture creates a small DataFrame with 2 rows containing all the columns
    used in racing data analysis. It includes race details, horse information, 
    performance metrics, and betting data. It's intended to be used across 
    multiple test files.
    
    Returns:
        pandas.DataFrame: A sample DataFrame with comprehensive racing data
    """
    data = {
        "Race PP ID": ["R20210101-01", "R20210101-02"],
        "Race ID": ["20210101-01", "20210101-02"],
        "Race Day": ["2021-01-01", "2021-01-01"],
        "Race Meeting Number": [1, 1],
        "Racecourse Code": ["TC01", "TC02"],
        "Racecourse Name": ["Tokyo", "Osaka"],
        "N-th Racing Day": [1, 1],
        "Race Condition": ["4yo+", "3yo+"],
        "Race Symbol/Drawing": ["", ""],
        "Race Symbol/Age": ["", ""],
        "Race Symbol/Mare": ["", ""],
        "Race Symbol/Sire": ["", ""],
        "Race Symbol/Special Weight": ["", ""],
        "Race Symbol/Mixed": ["", ""],
        "Race Symbol/Handicap": ["", ""],
        "Race Symbol/Drawing2": ["", ""],
        "Race Symbol/Market": ["", ""],
        "Race Symbol/Fixed Weight": ["", ""],
        "Race Symbol/Stallion": ["", ""],
        "Race Symbol/Kanto Distributed Horses": ["", ""],
        "Race Symbol/Specified": ["", ""],
        "Race Symbol/Kasai Distributed Horses": ["", ""],
        "Race Symbol/Horses from Kyushu": ["", ""],
        "Race Symbol/Apprentice": ["", ""],
        "Race Symbol/Gelding": ["", ""],
        "Race Symbol/International": ["", ""],
        "Race Symbol/Specified2": ["", ""],
        "Race Symbol/Special Specified": ["", ""],
        "Race Number": [1, 2],
        "Graded Races N-th Time": [0, 0],
        "Race Name": ["Tokyo Main Race", "Osaka Special"],
        "Listed and Graded Races": ["G1", "G2"],
        "Steeplechase Category": ["", ""],
        "Turf and Dirt Category": ["Turf", "Dirt"],
        "Turf and Dirt Category2": ["", ""],
        "Clockwise, Anti-clockwise and Straight Course Category": ["Right", "Left"],
        "Inner Circle, Outer Circle and Tasuki Course Category": ["Inner", "Outer"],
        "Distance(m)": [1600, 1800],
        "Weather": ["Sunny", "Cloudy"],
        "Track Condition1": ["Good", "Good"],
        "Track Condition2": ["", ""],
        "Post Time": ["15:45", "16:30"],
        "Final Position": [1, 2],
        "FP Note": ["", ""],
        "Bracket Number": [3, 5],
        "Post Position": [3, 1],
        "Horse Name": ["Super Horse", "Fast Runner"],
        "Sex": ["M", "F"],
        "Age": [4, 3],
        "Weight(Kg)": [55.0, 53.0],
        "Jockey": ["J. Smith", "T. Johnson"],
        "Total Time(1/10s)": [834, 856],
        "Margin": [0.5, 1.2],
        "Position 1st Corner": [2, 3],
        "Position 2nd Corner": [2, 2],
        "Position 3rd Corner": [1, 2],
        "Position 4th Corner": [1, 2],
        "Time of Last 3 Furlongs (600m)": [35.6, 36.2],
        "Win Odds(100Yen)": [2.1, 4.6],
        "Win Fav": [1, 3],
        "Horse Weight": [468, 452],
        "Horse Weight Gain and Loss": [2, -1],
        "East, West, Foreign Country and Local Category": ["East", "West"],
        "Trainer": ["T. Brown", "S. White"],
        "Owner": ["Racing Club A", "Racing Club B"],
        "Prize Money(10000Yen)": [5000, 2000],
        "year": [2021, 2021]
    }
    
    return pd.DataFrame(data)

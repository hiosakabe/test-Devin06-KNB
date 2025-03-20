"""
特徴量エンジニアリングを行うモジュール
"""
import pandas as pd
from tqdm import tqdm
from src.utils import Timer

# 発送前にわかっている情報のみ特徴量を絞る
FEATURE_COLUMNS = [
    'Race PP ID', 'Race ID', 'Race Day', 'Race Meeting Number',
    'Racecourse Code', 'Racecourse Name', 'N-th Racing Day',
    'Race Condition','Race Number', 'Race Name',
    'Listed and Graded Races', 'Steeplechase Category',
    'Turf and Dirt Category', 'Turf and Dirt Category2',
    'Clockwise, Anti-clockwise and Straight Course Category',
    'Inner Circle, Outer Circle and Tasuki Course Category', 'Distance(m)',
    'Weather', 'Track Condition1', 'Post Time',
    'FP Note', 'Bracket Number', 'Post Position', 'Horse Name', 'Sex',
    'Age', 'Weight(Kg)', 'Jockey', 'Margin',
    'Win Odds(100Yen)', 'Win Fav', 'Horse Weight',
    'Horse Weight Gain and Loss',
    'East, West, Foreign Country and Local Category', 'Trainer', 'Owner',
]

def create_numeric_features(input_df):
    """
    数値特徴量を抽出する関数（今回は全ての特徴量をそのまま利用）
    
    Parameters:
        input_df (pd.DataFrame): 入力のDataFrame
        
    Returns:
        pd.DataFrame: 数値特徴量のDataFrame
    """
    return input_df[FEATURE_COLUMNS].copy()

def generate_features(input_df):
    """
    複数の前処理関数を適用して特徴量を作成する関数
    
    Parameters:
        input_df (pd.DataFrame): 入力のDataFrame
        
    Returns:
        pd.DataFrame: 作成された特徴量のDataFrame
    """
    processing_functions = [create_numeric_features]
    combined_features = pd.DataFrame()
    
    for function in tqdm(processing_functions, total=len(processing_functions)):
        with Timer(prefix='create ' + function.__name__):
            features_part = function(input_df)
        assert len(features_part) == len(input_df), f"{function.__name__} で行数が一致しません"
        combined_features = pd.concat([combined_features, features_part], axis=1)
    return combined_features

"""
データ読み込みと前処理を行うモジュール
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_race_data(data_dir=None):
    """
    レース結果データを読み込む関数
    
    Parameters:
        data_dir (str): データディレクトリのパス
        
    Returns:
        pd.DataFrame: 結合されたレース結果データ
    """
    if data_dir is None:
        import os
        # リポジトリのルートディレクトリからの相対パス
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    
    data1 = pd.read_csv(f"{data_dir}/1986-1992_race_result.csv", low_memory=False)
    data2 = pd.read_csv(f"{data_dir}/1993-1999_race_result.csv", low_memory=False)
    data3 = pd.read_csv(f"{data_dir}/2000-2005_race_result.csv", low_memory=False)
    data4 = pd.read_csv(f"{data_dir}/2006-2009_race_result.csv", low_memory=False)
    data5 = pd.read_csv(f"{data_dir}/2010-2013_race_result.csv", low_memory=False)
    
    combined_data = pd.concat([data1, data2, data3, data4, data5])
    
    # 新しい列名を設定
    new_columns = [
        'Race PP ID', 'Race ID', 'Race Day', 'Race Meeting Number', 'Racecourse Code', 'Racecourse Name', 
        'N-th Racing Day', 'Race Condition', 'Race Symbol/Drawing', 'Race Symbol/Age', 'Race Symbol/Mare', 
        'Race Symbol/Sire', 'Race Symbol/Special Weight', 'Race Symbol/Mixed', 'Race Symbol/Handicap', 
        'Race Symbol/Drawing2', 'Race Symbol/Market', 'Race Symbol/Fixed Weight', 'Race Symbol/Stallion', 
        'Race Symbol/Kanto Distributed Horses', 'Race Symbol/Specified', 'Race Symbol/Kasai Distributed Horses', 
        'Race Symbol/Horses from Kyushu', 'Race Symbol/Apprentice', 'Race Symbol/Gelding', 'Race Symbol/International', 
        'Race Symbol/Specified2', 'Race Symbol/Special Specified', 'Race Number', 'Graded Races N-th Time', 
        'Race Name', 'Listed and Graded Races', 'Steeplechase Category', 'Turf and Dirt Category', 
        'Turf and Dirt Category2', 'Clockwise, Anti-clockwise and Straight Course Category', 
        'Inner Circle, Outer Circle and Tasuki Course Category', 'Distance(m)', 'Weather', 'Track Condition1', 
        'Track Condition2', 'Post Time', 'Final Position', 'FP Note', 'Bracket Number', 'Post Position', 
        'Horse Name', 'Sex', 'Age', 'Weight(Kg)', 'Jockey', 'Total Time(1/10s)', 'Margin', 
        'Position 1st Corner', 'Position 2nd Corner', 'Position 3rd Corner', 'Position 4th Corner', 
        'Time of Last 3 Furlongs (600m)', 'Win Odds(100Yen)', 'Win Fav', 'Horse Weight', 
        'Horse Weight Gain and Loss', 'East, West, Foreign Country and Local Category', 'Trainer', 
        'Owner', 'Prize Money(10000Yen)', 'year'
    ]
    combined_data.columns = new_columns
    
    return combined_data

def preprocess_data(data_df, target_column="Final Position"):
    """
    データの前処理を行う関数
    
    Parameters:
        data_df (pd.DataFrame): 入力データフレーム
        target_column (str): 目的変数のカラム名
        
    Returns:
        pd.DataFrame: 前処理済みのデータフレーム
    """
    # target_columnに欠損がない行のみ抽出
    non_null_indices = data_df[target_column].dropna(axis=0).index.tolist()
    filtered_df = data_df.iloc[non_null_indices]
    
    # 各列の型を確認し、object型の場合は欠損値を'N'で補完し、LabelEncoderで数値変換
    processed_df = filtered_df.copy()
    for column in processed_df.columns:
        if processed_df[column].dtype == 'object':
            processed_df[column] = processed_df[column].fillna('N')
            label_encoder = LabelEncoder()
            processed_df[column] = label_encoder.fit_transform(processed_df[column].values)
    
    # 残りの欠損値を持つ行を削除
    processed_df = processed_df.dropna(axis=0)
    
    return processed_df

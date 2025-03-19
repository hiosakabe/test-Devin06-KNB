"""
競馬レース結果予測のメイン実行ファイル
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_race_data, preprocess_data
from src.feature_engineering import generate_features, FEATURE_COLUMNS
from src.model import train_lgbm_model, visualize_predictions
from src.utils import Timer

def main():
    # データの読み込み
    print("データを読み込んでいます...")
    with Timer(prefix="データ読み込み時間"):
        data = load_race_data()
    
    # 目的変数の設定
    target_column = "Final Position"
    
    # データの前処理
    print("データを前処理しています...")
    with Timer(prefix="前処理時間"):
        processed_df = preprocess_data(data, target_column)
    
    # 特徴量とターゲットの分離
    features_df = processed_df[FEATURE_COLUMNS]
    target_series = processed_df[target_column]
    
    # 特徴量の生成
    print("特徴量を生成しています...")
    with Timer(prefix="特徴量生成時間"):
        train_features_df = generate_features(processed_df)
    
    # 欠損値の確認
    print("特徴量の欠損値を確認しています...")
    for col in train_features_df.columns:
        null_count = len(train_features_df[train_features_df[col].isnull()])
        if null_count > 0:
            print(f"{col}: {null_count}件の欠損値があります")
    
    # 欠損値を含む行を削除（元のノートブックと同じ方針）
    train_features_df = train_features_df.dropna(axis=0)
    print(f"欠損値削除後のデータ数: {len(train_features_df)}")
    
    # KFoldによる交差検証の設定
    print("モデルを学習しています...")
    kf = KFold(n_splits=5, shuffle=True, random_state=71)
    # 欠損値削除後のデータに合わせてtarget_arrayを更新
    target_series = target_series.loc[train_features_df.index]
    target_array = np.array(target_series)
    cv_indices = list(kf.split(train_features_df, target_array))
    
    # LightGBMのパラメータ（必要に応じて変更）
    lgbm_params = {}
    
    # モデルの学習を実行
    with Timer(prefix="モデル学習時間"):
        oof_preds, trained_models = train_lgbm_model(
            train_features_df.values, 
            target_array, 
            cv_indices, 
            params=lgbm_params, 
            verbose=100
        )
    
    # 予測値と実際の目的変数との散布図をプロット
    print("結果を可視化しています...")
    visualize_predictions(oof_preds, target_array, target_column)
    
    print("処理が完了しました。")

if __name__ == "__main__":
    main()

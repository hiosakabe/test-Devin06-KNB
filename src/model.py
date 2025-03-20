"""
モデル学習と評価を行うモジュール
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import lightgbm as lgbm
from src.utils import Timer

def train_lgbm_model(features, target, cv_splits, params: dict = None, verbose: int = 50):
    """
    LightGBMモデルの学習と交差検証の実施
    
    Parameters:
        features (np.ndarray): 特徴量
        target (np.ndarray): 目的変数
        cv_splits (list): 交差検証の分割リスト
        params (dict): LightGBMのパラメータ
        verbose (int): 学習中の詳細表示レベル
        
    Returns:
        tuple: (全体のout-of-fold予測, 学習済みモデルのリスト)
    """
    if params is None:
        params = {}

    model_list = []
    oof_predictions = np.zeros_like(target, dtype=np.float64)

    for fold_idx, (train_idx, valid_idx) in enumerate(cv_splits):
        X_train, y_train = features[train_idx], target[train_idx]
        X_valid, y_valid = features[valid_idx], target[valid_idx]

        model = lgbm.LGBMRegressor(**params)
        with Timer(prefix=f'fit fold={fold_idx}'):
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)]
                # early_stopping_rounds=100
            )

        fold_predictions = model.predict(X_valid)
        oof_predictions[valid_idx] = fold_predictions
        model_list.append(model)
        fold_rmsle = mean_squared_error(y_valid, fold_predictions) ** 0.5
        print(f'Fold {fold_idx} RMSLE: {fold_rmsle:.4f}\n')

    total_rmsle = mean_squared_error(target, oof_predictions) ** 0.5
    print('-' * 50)
    print(f'FINISHED | Whole RMSLE: {total_rmsle:.4f}')
    return oof_predictions, model_list

def visualize_predictions(predictions, actual, target_column_name):
    """
    予測値と実際の値の散布図を表示する関数
    
    Parameters:
        predictions (np.ndarray): 予測値
        actual (np.ndarray): 実際の値
        target_column_name (str): 目的変数の名前
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(target_column_name, fontsize=20)
    ax.set_xlabel('Predicted ' + target_column_name, fontsize=12)
    ax.set_ylabel('Actual ' + target_column_name, fontsize=12)
    ax.scatter(predictions, actual)
    plt.show()

# レーシングデータ分析

このリポジトリは、1986年から2021年までのレーシングデータを包括的に収集し、モータースポーツにおける統計分析とパフォーマンス追跡のために設計されています。

## 環境セットアップ

### 前提条件

- Python 3.12.8
- pyenv（Pythonバージョン管理用）
- Poetry（依存関係管理用）

### セットアップ手順

1. リポジトリをクローンする：
   ```bash
   git clone https://github.com/hiosakabe/test-Devin06-KNB.git
   cd test-Devin06-KNB
   ```

2. pyenvでPython環境をセットアップする：
   ```bash
   # Python 3.12.8がインストールされていない場合はインストール
   pyenv install 3.12.8
   
   # .python-versionファイルが自動的にPythonバージョンを設定
   # 以下のコマンドで確認：
   python --version  # Python 3.12.8と表示されるはず
   ```

3. Poetryで依存関係をインストールする：
   ```bash
   # Poetryがインストールされていない場合はインストール
   # curl -sSL https://install.python-poetry.org | python3 -
   
   # 依存関係をインストール
   poetry install
   
   # 仮想環境を有効化
   poetry shell
   ```

## プロジェクト構造

- `data/`：1986年から2021年までのレーシングデータを含むCSVファイル
- `notebook/`：探索的データ分析用のJupyterノートブック
- `src/`：データ処理と分析のためのPythonモジュール
  - `data_loader.py`：データ読み込みと前処理の関数
  - `feature_engineering.py`：特徴量作成と変換
  - `model.py`：機械学習モデルのトレーニングと評価
  - `utils.py`：ユーティリティ関数とクラス

## 使用例

```python
# 使用例
from src.data_loader import load_race_data, preprocess_data
from src.feature_engineering import generate_features
from src.model import train_lgbm_model
import numpy as np
from sklearn.model_selection import KFold

# データの読み込みと前処理
race_data = load_race_data()
processed_data = preprocess_data(race_data)

# 特徴量の生成
features = generate_features(processed_data)

# 目標変数の準備
target = processed_data["Final Position"].values

# 交差検証によるモデルトレーニング
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_splits = list(cv.split(features, target))
predictions, models = train_lgbm_model(features.values, target, cv_splits)
```

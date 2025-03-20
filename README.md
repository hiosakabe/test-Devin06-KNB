# Racing Data Analysis

This repository contains a comprehensive collection of racing data spanning from 1986 to 2021, designed for statistical analysis and performance tracking in motorsport.

## Environment Setup

### Prerequisites

- Python 3.12
- pyenv (for Python version management)
- Poetry (for dependency management)

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/hiosakabe/test-Devin06-KNB.git
   cd test-Devin06-KNB
   ```

2. Set up Python environment with pyenv:
   ```bash
   # Install Python 3.12.8 if not already installed
   pyenv install 3.12.8
   
   # The .python-version file will automatically set the Python version
   # Verify with:
   python --version  # Should show Python 3.12.8
   ```

3. Install dependencies with Poetry:
   ```bash
   # Install Poetry if not already installed
   # curl -sSL https://install.python-poetry.org | python3 -
   
   # Install dependencies
   poetry install
   
   # Activate the virtual environment
   poetry shell
   ```

## Project Structure

- `data/`: CSV files containing racing data from 1986 to 2021
- `notebook/`: Jupyter notebooks for exploratory data analysis
- `src/`: Python modules for data processing and analysis
  - `data_loader.py`: Functions for loading and preprocessing data
  - `feature_engineering.py`: Feature creation and transformation
  - `model.py`: Machine learning model training and evaluation
  - `utils.py`: Utility functions and classes

## Usage

```python
# Example usage
from src.data_loader import load_race_data, preprocess_data
from src.feature_engineering import generate_features
from src.model import train_lgbm_model
import numpy as np
from sklearn.model_selection import KFold

# Load and preprocess data
race_data = load_race_data()
processed_data = preprocess_data(race_data)

# Generate features
features = generate_features(processed_data)

# Prepare target variable
target = processed_data["Final Position"].values

# Train model with cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_splits = list(cv.split(features, target))
predictions, models = train_lgbm_model(features.values, target, cv_splits)
```

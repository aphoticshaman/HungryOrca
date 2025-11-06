# Hull Tactical - Market Prediction Solution

## Competition Overview
Predict S&P 500 excess returns while managing volatility constraints. Challenge the Efficient Market Hypothesis using machine learning.

**Key Constraints:**
- Volatility must stay within 120% of market volatility
- Allocation range: 0 to 2 (allowing some leverage)
- Metric: Sharpe ratio variant with penalties for excess volatility

## Project Structure
```
hull_tactical/
├── data/               # Competition data (download required)
├── models/            # Trained model artifacts
├── notebooks/         # Jupyter notebooks for exploration
├── src/              # Source code modules
└── README.md         # This file
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install pandas numpy scikit-learn xgboost lightgbm torch polars kaggle
```

### 2. Download Competition Data
You need to download the data from Kaggle. Two options:

**Option A: Using Kaggle CLI**
```bash
# Install kaggle CLI if not already installed
pip install kaggle

# Configure Kaggle API credentials
# Download your kaggle.json from https://www.kaggle.com/settings
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download competition data
cd hull_tactical/data
kaggle competitions download -c hull-tactical-market-prediction
unzip hull-tactical-market-prediction.zip
```

**Option B: Manual Download**
1. Go to https://www.kaggle.com/competitions/hull-tactical-market-prediction/data
2. Download train.csv and test.csv
3. Place them in `hull_tactical/data/`

### 3. Run the Solution
```bash
cd hull_tactical
python src/train_model.py
```

## Strategy Overview

### 1. Feature Engineering
- Technical indicators (RSI, MACD, Bollinger Bands)
- Momentum features
- Volatility measures (GARCH, realized volatility)
- Sentiment indicators
- Macroeconomic feature interactions
- Rolling statistics and regime detection

### 2. Model Ensemble
- XGBoost for capturing non-linear patterns
- LightGBM for speed and efficiency
- Neural Network for complex interactions
- Ridge/Lasso for linear baselines
- Weighted ensemble based on validation performance

### 3. Allocation Strategy
- Dynamic position sizing based on prediction confidence
- Volatility targeting to stay within constraints
- Kelly Criterion inspired allocation
- Drawdown protection mechanisms

### 4. Risk Management
- Real-time volatility monitoring
- Position limits based on market regime
- Stop-loss mechanisms
- Correlation-aware exposure management

## Key Innovations

1. **Regime Detection**: Identify bull/bear/sideways markets
2. **Adaptive Volatility**: Adjust allocation based on current vol regime
3. **Multi-Timeframe Analysis**: Combine signals from different horizons
4. **Feature Importance Tracking**: Monitor which features drive performance
5. **Online Learning**: Adapt to market changes during forecasting phase

## Expected Performance
- Target Sharpe Ratio: > 2.0
- Maximum Drawdown: < 15%
- Win Rate: > 55%
- Volatility: < 120% of market volatility

## Next Steps
1. Download data (see Setup Instructions)
2. Run exploratory analysis: `notebooks/01_eda.ipynb`
3. Train models: `python src/train_model.py`
4. Create submission: `notebooks/submission.ipynb`
5. Submit to Kaggle

## Competition Timeline
- Entry Deadline: December 8, 2025
- Final Submission: December 15, 2025
- Competition End: June 16, 2026

Let's challenge the Efficient Market Hypothesis and win this! 🚀

# Stock Price Prediction Analysis

## Overview
This project aims to predict stock prices using two approaches: ARIMA for time-series forecasting and Gradient Boosting for regression-based predictions. The workflow includes data preparation, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

## Data Preparation
### Data Source
- **Provider**: Yahoo Finance (`yfinance` library)
- **Stocks Analyzed**: AAPL, MSFT, GOOGL, AMZN, TSLA
- **Time Frame**: 2015-2025

### Libraries Used
- `pandas`, `numpy`: Data processing
- `yfinance`: Data acquisition
- `statsmodels`: ARIMA modeling
- `scikit-learn`, `xgboost`: Machine learning
- `matplotlib`, `seaborn`: Visualization

### Data Cleaning
- Missing values handled via forward fill.

## Exploratory Data Analysis (EDA)
- **Trends Analysis**: Historical closing prices
- **Moving Averages**: 20-day & 50-day rolling mean
- **Trading Volume**: Market activity insights

## Feature Engineering
- **Lagged Features**: Previous day's closing price
- **Rolling Mean**: 5-day rolling mean
- **Daily Price Change**: Percentage change computation

## Modeling Approaches
### ARIMA Model
- **Goal**: Time-series forecasting
- **Optimal Parameters**: `(p,d,q) = (2,1,1)`
- **Performance (Test RMSE)**: 189.200
- **Performance (Test MAE)**: 177.645
- **Performance (Test MAPE)**: 401.57%

### Gradient Boosting (XGBoost)
- **Goal**: Regression-based prediction
- **Best Parameters**: `{colsample_bytree: 0.8, learning_rate: 0.05, max_depth: 5, n_estimators: 200, subsample: 0.8}`
- **Performance (Test RMSE)**: 1.022
- **Performance (Test MAE)**: 0.60
- **Performance (Test MAPE)**: 0.64%

## Evaluation & Results
- **ARIMA**: Effective for short-term trends but weak overall.
- **XGBoost**: More accurate predictions with lower RMSE.
- **Visualization**: Actual vs. predicted stock prices.

## Recommendations
- **Hybrid Modeling**: Combining multiple approaches.
- **Additional Features**: RSI, MACD, volatility measures.
- **Optimization**: Bayesian tuning & regularization.

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas numpy yfinance statsmodels scikit-learn xgboost matplotlib seaborn

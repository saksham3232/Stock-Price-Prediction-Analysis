# Stock Price Prediction Analysis

## Overview
This project aims to predict stock prices using two distinct modeling approaches: the Autoregressive Integrated Moving Average (ARIMA) model for time-series forecasting and Gradient Boosting (via XGBoost) for regression-based predictions. The workflow encompasses data preparation, comprehensive exploratory data analysis (EDA), feature engineering, model training, and thorough evaluation, with a focus on Apple Inc. (AAPL) stock data. The objective is to identify the most effective predictive method to support trading strategies as of March 20, 2025.

## Data Preparation
### Data Source
- **Provider**: Yahoo Finance (accessed via the `yfinance` library)
- **Stocks Analyzed**: AAPL (Apple Inc.), MSFT (Microsoft Corp.), GOOGL (Alphabet Inc.), AMZN (Amazon Inc.), TSLA (Tesla Inc.)
- **Time Frame**: January 1, 2015, to January 1, 2025

### Libraries Used
- `pandas`, `numpy`: Data manipulation and numerical computations
- `yfinance`: Stock data acquisition from Yahoo Finance
- `statsmodels`: Implementation of ARIMA for time-series analysis
- `scikit-learn`, `xgboost`: Machine learning tools for splitting, evaluation, and Gradient Boosting
- `matplotlib`, `seaborn`: Visualization of trends, relationships, and model outcomes

### Data Cleaning
- Missing values were addressed using forward fill to ensure time-series continuity, preserving data integrity for modeling.

## Exploratory Data Analysis (EDA)
- **Trends Analysis**: Examined historical closing prices of AAPL to identify growth patterns and volatility over the decade.
- **Moving Averages**: Calculated 20-day and 50-day rolling means to smooth short-term fluctuations and highlight long-term trends.
- **Trading Volume**: Analyzed market activity and liquidity, noting volume spikes tied to significant price movements.
- **Correlation Heatmap**: Explored relationships among Open, High, Low, Close prices, and Volume, revealing strong price correlations and volume’s distinct role.
- **Pair Plots**: Visualized pairwise variable interactions, confirming linear price relationships and exploring volume dynamics.
- **Distribution of Closing Prices**: Assessed the right-skewed distribution of AAPL closing prices, indicating frequent lower values with occasional peaks.
- **Box Plot for Volume**: Examined trading volume spread and outliers, linking high-volume days to market events.
- **Rolling Statistics (Mean and Standard Deviation)**: Computed 30-day rolling mean and standard deviation to evaluate stationarity and volatility, showing non-stationary behavior.

## Feature Engineering
- **Lagged Features**: Included the previous day’s closing price (`Lag_1_Close`) to capture temporal dependencies.
- **Rolling Mean**: Derived a 5-day rolling mean (`Rolling_Mean_5`) to represent short-term trends.
- **Daily Price Change**: Calculated percentage change in closing prices (`Pct_Change_Close`) for volatility and momentum insights.
- **Data Integrity**: Dropped rows with missing values post-engineering to ensure a complete dataset.

## Modeling Approaches
### ARIMA Model
- **Goal**: Forecast stock prices using time-series properties, focusing on linear trends and seasonality.
- **Optimal Parameters**: `(p, d, q) = (2, 1, 1)`, determined through systematic tuning.
- **Performance (Test RMSE)**: 189.181, indicating large prediction errors.
- **Performance (Test MAE)**: 177.625, showing significant average deviations.
- **Performance (Test MAPE)**: 401.54%, reflecting poor relative accuracy.

### Gradient Boosting (XGBoost)
- **Goal**: Predict stock prices by capturing complex, non-linear patterns via regression.
- **Best Parameters**: `{colsample_bytree: 0.8, learning_rate: 0.05, max_depth: 5, n_estimators: 200, subsample: 0.8}`, optimized through tuning.
- **Performance (Test RMSE)**: 0.97, demonstrating minimal errors.
- **Performance (Test MAE)**: 0.50, indicating tight alignment with actual values.
- **Performance (Test MAPE)**: 0.55%, showcasing high relative accuracy.

## Evaluation & Results
- **ARIMA**: Effective for short-term trend modeling but struggles with non-linear market dynamics, resulting in high test errors.
- **XGBoost**: Delivers superior accuracy with significantly lower RMSE, MAE, and MAPE, adept at handling volatility and complex relationships.
- **Visualization**: Plots of actual vs. predicted prices highlight ARIMA’s divergence and XGBoost’s precision.

## Recommendations
- **Prioritize Gradient Boosting**: Leverage XGBoost for trading strategies due to its robust performance, ideal for short-term buy/sell decisions.
- **Enhanced Features**: Incorporate technical indicators (e.g., RSI, MACD), volatility measures, and sentiment data from sources like [StockTwits](https://stocktwits.com/) to refine predictions.
- **Model Optimization**: Apply regularization (L1, L2) to prevent overfitting and use Bayesian optimization for efficient hyperparameter tuning.

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas numpy yfinance statsmodels scikit-learn xgboost matplotlib seaborn

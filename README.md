# Stock Market Prediction using ARIMA and Gradient Boosting Models - README

## Overview

This project focuses on predicting stock market prices using two distinct methods:

*   **ARIMA (Autoregressive Integrated Moving Average):** A time-series analysis technique to capture temporal dependencies within the stock price data.
*   **Gradient Boosting:** A machine learning algorithm that leverages multiple weak learners to create a strong predictive model.

The workflow encompasses several key stages: data preparation, exploratory data analysis (EDA), feature engineering, model building, hyperparameter tuning, and model evaluation.

## Project Structure

The project is structured into the following steps:

1.  **Data Preparation:**
    *   **Import Libraries:** Loading essential Python libraries for data manipulation (Pandas, NumPy), stock data retrieval (`yfinance`), time series analysis (`statsmodels`), visualization (`matplotlib`, `seaborn`), and machine learning (`scikit-learn`, `xgboost`).
    *   **Download Data:** Fetching historical stock data (Open, High, Low, Close, Volume) for specified tickers (AAPL, MSFT, GOOGL, AMZN, TSLA) from Yahoo Finance between "2015-01-01" and "2025-01-01". Saving the Apple stock data as a CSV file ("Apple\_Data.csv").
    *   **Handle Missing Values:** Addressing missing data points using forward fill (`ffill`) to maintain data continuity.

2.  **Exploratory Data Analysis (EDA):**
    *   **Description:** Calculating descriptive statistics (mean, standard deviation, quartiles) to understand the data distribution.
    *   **Info:** Displaying the data types and non-null counts for each column.
    *   **Visualize Trends:** Plotting the closing prices over time to visually identify trends.
    *   **Moving Averages:** Calculating and visualizing 20-day and 50-day rolling means to smooth short-term fluctuations.
    *   **Visualize Volume Trends:** Plotting trading volume over time to analyze trading activity.
    *   **Correlation Heatmap:** Displaying the correlation matrix between different features to assess relationships.
    *   **Pair Plots:** Visualizing pairwise relationships between variables in the dataset.
    *   **Distribution of Closing Prices:** Plotting a histogram and KDE to understand the distribution of closing prices.
    *   **Box Plot for Volume:** Identifying outliers in trading volume using a box plot.
    *   **Rolling Statistics (Mean and Standard Deviation):** Calculating and visualizing rolling mean and standard deviation to assess time series stationarity.

3.  **Feature Engineering:**
    *   **Create Features:** Generating new features to improve model performance:
        *   Lagged closing price (previous day's closing price).
        *   5-day rolling mean of closing prices.
        *   Percentage change in closing price (daily returns).
    *   Removing rows with any missing values resulting from feature engineering.

4.  **Modeling:**

    *   **ARIMA Model with Hyperparameter Tuning:**
        *   Defining a "safe\_mape" function to handle edge cases in MAPE calculation
        *   Defining a "check\_stationarity" function to check if the time series is stationary
        *   Splitting the data into training and testing sets (80% train, 20% test).
        *   Defining a function "optimize\_arima" to identify the best parameters for the ARIMA model
        *   Fitting and forecasting with ARIMA
        *   Evaluating the model performance using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE).
    *   **ARIMA Model with Fixed Parameters:**
        *   Setting fixed parameters (p=2, d=1, q=2) for the ARIMA model.
        *   Fitting the ARIMA model.
        *   Forecasting the next 10 steps.
        *   Calculating performance metrics (RMSE, MAE, MAPE, R-squared, MASE).
        *   Comparing the R-squared value to a simple benchmark model (mean).

    *   **Gradient Boosting Model with Hyperparameter Tuning:**
        *   Defining features (Lag\_1\_Close, Rolling\_Mean\_5, Pct\_Change\_Close) and target variable (Close).
        *   Splitting data into training and testing sets.
        *   Setting up a parameter grid for hyperparameter tuning.
        *   Using RandomizedSearchCV for hyperparameter optimization.
        *   Training the XGBoost model.
        *   Evaluating model performance on the test set.

## Model Evaluation Metrics

The following metrics are used to evaluate the performance of the models:

*   **Root Mean Squared Error (RMSE):** Measures the average magnitude of the errors. Lower values indicate better performance.
*   **Mean Absolute Error (MAE):** Measures the average absolute difference between predicted and actual values. Lower values indicate better performance.
*   **Mean Absolute Percentage Error (MAPE):** Measures the average percentage difference between predicted and actual values. Lower values indicate better performance.
*   **R-squared (R²):** Measures the proportion of variance in the dependent variable that can be predicted from the independent variables. Higher values (closer to 1) indicate a better fit.
*   **Mean Absolute Scaled Error (MASE):** Measures the forecast accuracy compared to a naive forecast. Values less than 1 indicate better performance than the naive forecast.

## Results

*   **ARIMA with Hyperparameter Tuning:**
    *   Best ARIMA Parameters: (2, 1, 2)
    *   ARIMA RMSE: 62.8092
    *   ARIMA MAE: 56.6089
    *   ARIMA MAPE: 28.38%
*   **ARIMA with Fixed Parameters:**
    *   Fixed Parameters for ARIMA: p=2, d=1, q=2
    *   ARIMA RMSE: 4.5548
    *   ARIMA MAE: 3.7653
    *   ARIMA MAPE: 1.47%
    *   ARIMA R²: -0.8008 (Poor fit)
    *   ARIMA MASE: 3.1219
*   **Gradient Boosting Model with Hyperparameter Tuning:**

    *   Best parameters: {'subsample': 0.5, 'min\_child\_weight': 5, 'max\_depth': 30, 'learning\_rate': 0.1, 'gamma': 0.1, 'colsample\_bytree': 1}
    *   Tuned XGBoost RMSE: 0.8938
    *   Tuned XGBoost MAE: 0.4847
    *   Tuned XGBoost MAPE: 0.57%
    *   Tuned XGBoost R^2: 0.9998

## Conclusion

This project demonstrated the application of ARIMA and Gradient Boosting models for stock market price prediction. The Gradient Boosting model with hyperparameter tuning outperformed the ARIMA models, achieving a high R-squared value and low error metrics. The results highlight the potential of machine learning techniques for financial forecasting.
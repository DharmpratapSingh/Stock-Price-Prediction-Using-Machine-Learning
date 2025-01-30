# Stock-Price-Prediction-Using-Machine-Learning

## Overview

This project predicts NVIDIA (NVDA) stock prices using historical data and machine learning models. By engineering financial features and training various models, the project demonstrates the power of data analytics and machine learning in stock price forecasting.

## Objectives
	•	Predict future stock prices for NVIDIA based on historical trends.
	•	Engineer financial features like moving averages and daily returns to improve model accuracy.
	•	Evaluate and compare multiple machine learning models, including Linear Regression, Random Forest, and XGBoost.
	•	Visualize model predictions and feature importance for better interpretability.

## Tools & Technologies
	•	Programming Language: Python
	•	Data Manipulation: pandas, NumPy
	•	Visualization: Matplotlib, Seaborn
	•	Machine Learning Models: scikit-learn, XGBoost
	•	Data Source: yfinance

## Dataset
	•	Historical stock prices for NVIDIA (NVDA) were fetched using the Yahoo Finance API.
	•	The dataset includes:
	•	Daily Open, High, Low, Close, and Volume prices.
	•	Engineered features:
	•	Daily returns (percentage change in closing price).
	•	10-day and 50-day moving averages (average closing prices over the last 10 and 50 days, respectively).

## Methodology
	1.	Data Collection: Downloaded NVIDIA stock data (2018–2024) using the Yahoo Finance API.
	2.	Feature Engineering:
	  •	Calculated daily returns to capture stock volatility.
	  •	Created 10-day and 50-day moving averages to identify short-term and long-term trends.
	3.	Model Training & Evaluation:
	  •	Split data into training (80%) and testing (20%) sets.
	  •	Trained three machine learning models: Linear Regression, Random Forest, and XGBoost.
	  •	Evaluated models using Mean Squared Error (MSE) and R² Score.
	4.	Visualization:
	  •	Plotted Actual vs. Predicted Prices for each model.
	  •	Visualized Feature Importance using Random Forest.

## Results
	•	Random Forest Model:
	  •	Achieved an R² score of 0.997, showcasing its ability to predict stock prices with high accuracy.
	•	  Outperformed Linear Regression and XGBoost in terms of both MSE and R².
	•	Feature Importance:
	  •	The Close price was the most influential predictor, followed by moving averages.

## Key Visualizations
	1.	Actual vs. Predicted Prices (Random Forest Model):
    Demonstrates the accuracy of the model over the test dataset.
	2.	Feature Importance Chart:
    Highlights the contribution of each feature (e.g., Close Price, Returns) to the prediction.

## Future Enhancements
	•	Add advanced technical indicators like:
	•	Relative Strength Index (RSI)
	•	Bollinger Bands
	•	Perform sentiment analysis on NVIDIA news headlines to evaluate market impact.
	•	Extend the project to compare NVIDIA’s stock performance with competitors like AMD and TSM.

## Conclusion

This project showcases how machine learning and data engineering can predict stock prices accurately. The strong performance of the Random Forest model highlights its utility in financial forecasting, paving the way for further enhancements in stock price prediction.

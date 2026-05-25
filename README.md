# [Project 1: Loan Approval Prediction & Clustering System](https://github.com/Back-number212/loan_approval_ml)

- Developed an end-to-end data pipeline to automate banking credit risk assessment and optimize loan approval decisions using a historical financial dataset of 4,269 applicants. The preprocessing phase involved comprehensive Exploratory Data Analysis (EDA), feature engineering, binary label encoding for categorical variables, and numerical scaling using MinMaxScaler to ensure structural data balance.
- For supervised learning, 29 classification models were benchmarked using LazyPredict before selecting and fine-tuning LightGBM and XGBoost to predict loan approval statuses. LightGBM delivered top-tier performance, achieving an exceptional 98.9% accuracy. Feature importance analysis revealed that an applicant's credit score (cibil_score) acts as the single most critical dominant factor in approval decisions, heavily outweighing physical asset valuations.
- To uncover deeper customer behaviors, PCA was deployed for dimensionality reduction alongside unsupervised clustering frameworks. Implementing K-Means (optimized via the Elbow Method) effectively segmented the applicant pool into two distinct consumer wealth tiers, while DBSCAN density analysis successfully isolated 28 high-risk financial anomalies and noise points to mitigate institutional lending risks.
- Risk Mitigation: Successfully grouped applicants into 2 distinct wealth tiers and isolated 28 high-risk financial outliers (noise points) via DBSCAN.

# [Project 2: Stocks Price Prediction Using RNN](https://github.com/Back-number212/Stocks_Price_Prediction_Using_RNN/tree/main) 

- Implemented a data pipeline including Min-Max Normalization and Windowing techniques to process 5 years of historical data,ensuring optimal input for time-series forecasting.
- Developed a deep learning system using SimpleRNN, LSTM, and ARIMA models to predict closing prices.
- Evaluated model performance and visualization, achieving high precision with low RMSE in capturing market trends and providing data-driven insights for investment strategies.


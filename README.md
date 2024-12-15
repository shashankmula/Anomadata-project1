# Anomadata-project1
Automated Anomaly Detection for Predictive Maintenance

Project Overview: AnomaData - Automated Anomaly Detection for Predictive Maintenance This project focuses on leveraging machine learning techniques to automate anomaly detection for predictive maintenance in industrial systems. By analyzing sensor data over time, the goal is to identify unusual patterns or anomalies that could indicate impending failures or maintenance needs. This proactive approach helps optimize maintenance schedules, reduce downtime, and prevent costly breakdowns.

Key Steps and Methodology Data Collection and Preprocessing:

Dataset: The project uses a time-series dataset that is provided in an Excel file, which contains various sensor readings along with a target variable (y) indicating the presence of anomalies (1 for anomaly, 0 for normal). Data Cleaning: Redundant columns are removed, and missing values are handled using backward filling. The time column is converted into a datetime format and set as the index to enable time-series analysis. Exploratory Data Analysis (EDA):

The distribution of the target variable (y) is visualized using count plots. Anomalies over time are analyzed by resampling the data on an hourly basis to understand their patterns and distribution. Feature Engineering:

Temporal features are extracted from the time index, such as hour, day, and weekday, to capture the time-dependent patterns in the data. Rolling statistics (mean) and differences of key columns are added as additional features to capture trends and volatility in sensor readings. Model Preparation:

The data is split into predictors (X) and the target variable (y). Standard scaling is applied to the numerical features for better model performance. The dataset is split into training and test sets using an 80/20 ratio, with stratification to preserve the distribution of anomalies. Handling Imbalanced Data:

To address the class imbalance (i.e., anomalies are rare), the Synthetic Minority Over-sampling Technique (SMOTE) is used to oversample the minority class (anomalies) in the training set, ensuring that the model learns from a balanced dataset. Model Training and Evaluation: Several classification models are trained and evaluated, including:

Model Training
1) Two algorithms used: KNN Classifier and Random Forest.
2) The corresponding hyper parameters are stored in model.yaml
3) Using neuro-mf package, the best model giving the best scores with the given parameters are chosen and the model is stored as in the artifact folder as model.pkl

Model Evaluation
The score of the model present in the S3 bucket is matched against the model trained currently. If the change in score is above the threshold mentioned in schema file, i.e, if the current model has better score than the model in S3 bucket and the difference is above the threshold, then evaluation status is set to True. The current model will be now pushed to the S3 bucket in the next stage.
If there is no model in the S3, then the current model will be pushed into it, provided it is above the trained model threshold score.

The best-performing model is saved using the pickle library, allowing for easy deployment and future inference. Summary of Results Best Model: Random Forest Classifier (RF) Precision: RF outperforms others slightly in terms of precision, making it more reliable for detecting true positives (anomalies). ROC-AUC: XGBoost achieved a slightly higher AUC, but RF still provides excellent discriminatory power. Conclusion: RF is chosen for its precision and overall performance in predicting anomalies. Deployment The trained Random Forest model is saved into a file (anomaly_detection_model.pkl) using joblib for deployment. This allows the model to be easily loaded and used for making predictions on new, unseen data.

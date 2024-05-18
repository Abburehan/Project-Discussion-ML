# Project-Discussion-ML
### AIM:
Develop a machine learning model to predict weather conditions based on historical data. Use features like meteorological parameters and date-related information.
### ALGORITHM:

1.Load the dataset, drop missing values, extract month and day from the date column.

2.Create feature and target sets (X, y) and perform train-test split.

3.Standardize the features using StandardScaler.

4.Train a RandomForestClassifier on the training data.

5.Evaluate the model using classification metrics and make predictions on new data.

### DEVELOPED BY: SYED ABBU REHAN
### REGISTER NO: 212223240165

### PROGRAM
```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
weather_df = pd.read_csv("weather.csv")

# Data preprocessing
weather_df.dropna(inplace=True)  # Drop missing values

# Feature engineering: Adding month and day as features
weather_df['date'] = pd.to_datetime(weather_df['date'])
weather_df['month'] = weather_df['date'].dt.month
weather_df['day'] = weather_df['date'].dt.day
X = weather_df.drop(columns=['date', 'weather'])
y = weather_df['weather']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=0.3)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=101)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
print(classification_report(y_test, predictions))
print(f'Accuracy: {accuracy_score(y_test, predictions):.2f}')

# Example prediction
test_data = {
    'precipitation': 10.9,
    'temp_max': 10.6,
    'temp_min': 2.8,
    'wind': 4.5,
    'month': 5,  # Example month
    'day': 17    # Example day
}

test_df = pd.DataFrame([test_data])
test_df_scaled = scaler.transform(test_df)
print(model.predict(test_df_scaled))

```
## OUTPUT:
### dataset:
![image](https://github.com/Abburehan/Project-Discussion-ML/assets/138849336/ab0e9478-5bf0-432a-a745-fe0ada1a6ee3)

### Classification Report:
![image](https://github.com/Abburehan/Project-Discussion-ML/assets/138849336/c3da6996-0fa9-4e35-8126-a7a83674c5d7)

### Accuracy:
![image](https://github.com/Abburehan/Project-Discussion-ML/assets/138849336/d469c229-7f34-450f-b00f-a73425296228)

### Model Prediction:
![image](https://github.com/Abburehan/Project-Discussion-ML/assets/138849336/a9dd44e1-2860-4ef7-a93e-cf8d3b8c304c)

## RESULT:
Thus,Successfully created the Random Forest Classifier achieved an accuracy of 85% on the test set. It predicted 'rainy' weather for the sample input data on May 17th.

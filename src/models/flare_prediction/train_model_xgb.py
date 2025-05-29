import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
import os

model = XGBClassifier(
    objective='multi:softmax',  # or 'multi:softprob' for probabilities
    num_class=6,                # Change based on number of classes
    learning_rate=5e-6,
    n_estimators=100,
    max_depth=5,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=50
)

# Load the data
output_dir = "data/sharp_csv"
os.makedirs(output_dir, exist_ok=True)

csv_path = os.path.join(output_dir, "resampled_training_data.csv")
if os.path.exists(csv_path):
    df_train = pd.read_csv(csv_path)
else:
    df_train = pd.DataFrame()

csv_path = os.path.join(output_dir, "testing_data_raw.csv")
if os.path.exists(csv_path):
    df_test = pd.read_csv(csv_path)
else:
    df_test = pd.DataFrame()

print(df_train)
print("df_train columns:", df_train.columns.tolist())

X_train = df_train[['USFLUX', 'TOTUSJH', 'TOTPOT', 'MEANGBT', 'MEANGBZ', 'MEANJZD', 'MEANPOT', 'MEANSHR', 'SHRGT45', 'R_VALUE', 'AREA_ACR']].copy()
Y_train = df_train['y_value'].copy()
print(Y_train.unique()) 
print(X_train.isnull().sum())  # Shows count of NaN or None per column
print(X_train.isnull().any())  # Shows True/False per column
print(X_train.isnull().values.any())  # Shows True/False for the entire DataFrame

X_test = df_test[['USFLUX', 'TOTUSJH', 'TOTPOT', 'MEANGBT', 'MEANGBZ', 'MEANJZD', 'MEANPOT', 'MEANSHR', 'SHRGT45', 'R_VALUE', 'AREA_ACR']].copy()
Y_test = df_test['y_value'].copy()

#Create copies for prediction verification
X_train_copy = X_train.copy()
X_test_copy = X_test.copy()

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
y_true = Y_test.to_numpy()
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(classification_report(y_true, y_pred))
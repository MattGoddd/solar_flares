import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, softmax

import os

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

tf.random.set_seed(1234)
model = Sequential(
    [
        tf.keras.Input(shape = (X_train.shape[1],)),
        Dense(64, activation=relu),
        Dense(48, activation=relu),
        Dense(32, activation=relu),
        Dense(16, activation=relu),
        Dense(8, activation=relu),
        Dense(4, activation=relu),
        Dense(6, activation = linear),
    ], name = "flare_predictor"
)

# model.summary()

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-6, clipnorm = 1.0),
    metrics = ['accuracy', 'sparse_categorical_accuracy'],
)

history = model.fit(
    X_train, Y_train,
    batch_size = 64,
    epochs=25,
    validation_data = (X_test, Y_test),
)

history.history

results = model.evaluate(X_test, Y_test, batch_size=64)
print(results)

probabilities = model.predict(X_test)
y_hat = np.argmax(probabilities, axis=1)
y_true = Y_test.to_numpy()
print(f"y_hat: {y_hat}")
print(f"y_true: {y_true}")

precision = precision_score(y_true, y_hat, average='weighted')  # or 'macro', 'micro'
recall = recall_score(y_true, y_hat, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Optional: full report
print(classification_report(y_true, y_hat))
    
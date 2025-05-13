import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, softmax

import os

output_dir = "data/sharp_csv"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "training_data.csv")

if os.path.exists(csv_path):
    df_train = pd.read_csv(csv_path)
else:
    df_train = pd.DataFrame()

csv_path = os.path.join(output_dir, "testing_data.csv")
if os.path.exists(csv_path):
    df_test = pd.read_csv(csv_path)
else:
    df_test = pd.DataFrame()

X_train = df_train[['USFLUX', 'TOTUSJH', 'TOTPOT', 'MEANGBT', 'MEANGBZ', 'MEANJZD', 'MEANPOT', 'MEANSHR', 'SHRGT45', 'R_VALUE', 'AREA_ACR']].copy()
Y_train = df_train['y_value'].copy()
print(Y_train.unique()) 
print(X_train.isnull().sum())  # Shows count of NaN or None per column
print(X_train.isnull().any())  # Shows True/False per column
print(X_train.isnull().values.any())  # Shows True/False for the entire DataFrame

X_test = df_test[['USFLUX', 'TOTUSJH', 'TOTPOT', 'MEANGBT', 'MEANGBZ', 'MEANJZD', 'MEANPOT', 'MEANSHR', 'SHRGT45', 'R_VALUE', 'AREA_ACR']].copy()
Y_test = df_test['y_value'].copy()

tf.random.set_seed(1234)
# model = Sequential(
#     [
#         tf.keras.Input(shape = (X_train.shape[1],)),
#         Dense(64, activation=relu),
#         Dense(48, activation=relu),
#         Dense(32, activation=relu),
#         Dense(16, activation=relu),
#         Dense(8, activation=relu),
#         Dense(4, activation=relu),
#         Dense(7, activation = linear),
#     ], name = "flare_predictor"
# )

# # model.summary()

# model.compile(
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.5e-6),
# )

# history = model.fit(
#     X_train, Y_train,
#     batch_size = 64,
#     epochs=40,
#     validation_data = (X_test, Y_test),
# )

# history.history

# results = model.evaluate(X_test, Y_test, batch_size=64)
# print(results)
# predictions = model.predict(X_test[:3])
# print("predictions shape:", X_test[:3], predictions)


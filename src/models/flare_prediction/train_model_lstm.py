import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.activations import linear, relu, softmax, tanh
import os

def build_model():
    """
    Constructs a machine learning model based on the specified type.

    Returns:
        model: A compiled Keras model.
    """

    tf.random.set_seed(1234)

    model = Sequential(
    [
        tf.keras.Input(shape = (1, X_train.shape[1])),
        LSTM(64, activation=tanh, dropout = 0.2, return_sequences=True),
        LSTM(48, activation=tanh, dropout = 0.2,return_sequences=True),
        LSTM(32, activation=tanh, dropout = 0.2, return_sequences=True),
        LSTM(16, activation=tanh, dropout = 0.2, return_sequences=True),
        LSTM(8, activation=tanh, dropout = 0.2, return_sequences=True),
        LSTM(6, activation = linear),
    ], name = "flare_predictor"
        )

    return model

def compile_model(model, learning_rate = 5e-6, clipnorm = 1.0):
    """
    Compiles the model with specified loss function, optimizer, and metrics.

    Args:
        model: The Keras model to compile.
        learning_rate (float): Learning rate for the optimizer.
        clipnorm (float): Clipnorm for the optimizer.

    Returns:
        model: The compiled Keras model.
    """

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm = clipnorm),
        metrics = ['accuracy', 'sparse_categorical_accuracy'],
    )

    return model

def train_model(model, X_train, Y_train, X_test, Y_test, batch_size = 64, epochs = 25):
    """
    Trains the model on the training data and evaluates it on the test data.

    Args:
        model: The Keras model to train.
        X_train: Training features.
        Y_train: Training labels.
        X_test: Test features.
        Y_test: Test labels.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train.

    Returns:
        history: Training history.
    """

    history = model.fit(
        X_train, Y_train,
        batch_size = batch_size,
        epochs=epochs,
        validation_data = (X_test, Y_test),
    )

    return history

def evaluate_model(model, X_test, Y_test, batch_size = 64):
    """
    Evaluates the model on the test data.

    Args:
        model: The Keras model to evaluate.
        X_test: Test features.
        Y_test: Test labels.
        batch_size (int): Batch size for evaluation.

    Returns:
        results: Evaluation results.
    """

    results = model.evaluate(X_test, Y_test, batch_size=batch_size)
    probabilities = model.predict(X_test)
    y_hat = np.argmax(probabilities, axis=1)
    y_true = Y_test.to_numpy()
    precision = precision_score(y_true, y_hat, average='weighted')
    recall = recall_score(y_true, y_hat, average='weighted')
    report = classification_report(y_true, y_hat)

    return results, precision, recall, report
    
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

model = build_model("lstm")
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))  
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))      

model = compile_model(model)
train_model(model, X_train, Y_train, X_test, Y_test)
results, precision, recall, report = evaluate_model(model, X_test, Y_test)
print(f"Results: {results}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(report)
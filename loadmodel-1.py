import importlib.util
from openfl.interface.model import get_model
import joblib
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def to_sequence(x, y, seq_size=1):
    """Convert data into sequence for LSTM input"""
    x_values = []
    y_values = []
    for i in range(len(x) - seq_size):
        x_values.append(x.iloc[i:(i + seq_size)].values)
        y_values.append(y.iloc[i + seq_size])
    return np.array(x_values), np.array(y_values)

def Calculate(df, train_size, client_name):
    """Calculate and visualize model performance"""
    # Process timestamp data
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    # Select relevant columns
    feature_names = [
        'jetson_vdd_cpu_gpu_cv_mw',
        'jetson_gpu_usage_percent',
        'jetson_board_temperature_celsius',
        'jetson_vdd_in_mw',
        'jetson_cpu_usage_percent',
        'jetson_ram_usage_mb',
        'node_network_receive_bytes_total_KBps',
        'node_network_transmit_bytes_total_KBps'
    ]
    df = df[feature_names]

    # Replace zeros with small value to avoid division issues
    df.replace(0, 0.01, inplace=True)

    # Split into train and test sets
    split_index = int(len(df) * train_size)
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    # Create sequences
    seq_size = 40
    X_train, y_train = to_sequence(train, train, seq_size)
    X_test, y_test = to_sequence(test, test, seq_size)

    # Training set predictions
    trainPredict = model.predict(X_train)
    trainMAE = np.mean(np.abs(trainPredict - X_train), axis=1)
    trainMAPE = np.mean(np.abs(trainPredict - X_train) / X_train, axis=1) * 100

    # Individual histograms for training MAPE with feature names
    plt.figure(figsize=(15, 10))
    for i in range(trainMAPE.shape[1]):
        plt.subplot(4, 2, i + 1)  # Create a grid layout (4 rows, 2 columns)
        plt.hist(trainMAPE[:, i], bins=30, alpha=0.7, color='green')
        plt.title(f'MAPE for {feature_names[i]}')
        plt.xlabel('Mean Absolute Percentage Error (MAPE)')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f'{client_name}_train_mape_histogram.png')
    plt.close()

    # Print training results
    print(f"\n{client_name} Results:")
    print("Mean of Train MAE:", np.mean(trainMAE))
    print("Mean of Train MAPE:", np.mean(trainMAPE))
    print("Max of Train MAPE:", max(trainMAPE))

    # 1. Plot training predictions vs actual values
    plt.figure(figsize=(15, 6))
    plt.plot(X_train.flatten()[:200], label='Actual', color='blue', alpha=0.7)
    plt.plot(trainPredict.flatten()[:200], label='Predicted', color='red', alpha=0.7)
    plt.title(f'{client_name} Training Data: Actual vs Predicted')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{client_name}_train_comparison.png')
    plt.close()

    # Test set predictions
    testPredict = model.predict(X_test)
    testMAE = np.mean(np.abs(testPredict - X_test), axis=1)
    testMAPE = np.mean(np.abs(testPredict - X_test) / X_test, axis=1) * 100

    # Individual histograms for test MAPE with feature names
    plt.figure(figsize=(15, 10))
    for i in range(testMAPE.shape[1]):
        plt.subplot(4, 2, i + 1)  # Create a grid layout (4 rows, 2 columns)
        plt.hist(testMAPE[:, i], bins=30, alpha=0.7, color='green')
        plt.title(f'MAPE for {feature_names[i]}')
        plt.xlabel('Mean Absolute Percentage Error (MAPE)')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f'{client_name}_test_mape_histogram.png')
    plt.close()

    # Print test results
    print("Mean of Test MAE:", np.mean(testMAE))
    print("Mean of Test MAPE:", np.mean(testMAPE))
    print("Max of Test MAPE:", max(testMAPE))

    # 2. Plot test MAPE distribution
    plt.figure(figsize=(10, 6))
    plt.hist(testMAPE, bins=30, alpha=0.7)
    plt.axvline(np.mean(testMAPE), color='r', linestyle='dashed', linewidth=2, label=f'Mean MAPE: {np.mean(testMAPE):.2f}%')
    plt.xlabel('MAPE (%)')
    plt.ylabel('Frequency')
    plt.title(f'{client_name} Testing MAPE Distribution')
    plt.legend()
    plt.savefig(f'{client_name}_test_mape.png')
    plt.close()

    # 3. Plot test MAE timeline
    plt.figure(figsize=(15, 6))
    plt.plot(testMAE, color='red', alpha=0.7)
    plt.axhline(np.mean(testMAE), color='blue', linestyle='dashed', linewidth=2, label=f'Mean MAE: {np.mean(testMAE):.4f}')
    plt.title(f'{client_name} Testing MAE Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{client_name}_test_mae_timeline.png')
    plt.close()

# Load model and data


model = keras.models.load_model("output_model", compile=False)
#model = keras.models.load_model("models-611/lstm-100-64", compile=False)
#model = keras.models.load_model("models-611/lstm-64-64", compile=False)
#model = keras.models.load_model("models-611/lstm-100-128", compile=False)


df2 = pd.read_csv('data/Anomalous_data/Nano07.csv')
df1 = pd.read_csv('data/Anomalous_data/Nano08.csv')

train_size = 0.8
Calculate(df1, train_size, "Client_1")
Calculate(df2, train_size, "Client_2")
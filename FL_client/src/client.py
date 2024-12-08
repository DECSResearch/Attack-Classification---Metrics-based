# Muhammad Hamza Karim

import os
import joblib
import argparse
import flwr as fl
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import keras as ke
from typing import Tuple
import warnings
from keras import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Bidirectional
import matplotlib.pyplot as plt

warnings.simplefilter('ignore')

# Limit TensorFlow memory usage to 1 GB
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    except RuntimeError as e:
        print(e)


# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--ip', help='Provide the IP address', default="0.0.0.0", required=False)
parser.add_argument('--port', help='Provide the Port address', default="8080", required=False)
parser.add_argument('--id', help='Provide the client id', default="1", required=True)
parser.add_argument('--folder', help='Provide the Dataset folder', default='Client_1_RF', type=str, required=False)
args = parser.parse_args()

# Constants
SERVER_ADDR = f"{args.ip}:{args.port}"
FOLDER_LOC = args.folder
CLIENT_ID = args.id

temp_loss = []
temp_mape = []

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# Load Dataset Function
# def load_dataset():
#     folder_path = os.path.join('.', 'Train_data', FOLDER_LOC)
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.csv'):
#             file_path = os.path.join(folder_path, filename)
#             dataframe = pd.read_csv(file_path)
#             dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp']) # Convert 'datetimestamp' column to datetime
#             df = dataframe[['timestamp', 'jetson_vdd_cpu_gpu_cv_mw', 'jetson_gpu_usage_percent',
#                             'jetson_board_temperature_celsius', 'jetson_vdd_in_mw', 'jetson_cpu_usage_percent',
#                             'jetson_ram_usage_mb', 'node_network_receive_bytes_total_KBps',
#                             'node_network_transmit_bytes_total_KBps']]
#             df.set_index('timestamp', inplace=True)  # Set 'datetimestamp' as index
#             df.replace(0, 0.01, inplace=True)
#             # Count the number of rows where zero values were replaced with 0.01
#             num_rows_with_zero_replaced = len(df[(df == 0.01).any(axis=1)])
#             print(f"Number of rows where zero values were replaced: {num_rows_with_zero_replaced}")
#
#             # check if there are any remaining zero values
#             remaining_zeros = df[(df == 0).any(axis=1)]
#             print(f"Remaining rows with zero values: {len(remaining_zeros)}")
#     print("First few rows of the DataFrame:")
#     print(df.head())
#     print("Column names:")
#     print(df.columns)
#     return df

def load_dataset():
    folder_path = os.path.join('.', 'Train_data', FOLDER_LOC)

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            dataframe = pd.read_csv(file_path)

            # Convert 'timestamp' column to datetime
            dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'])

            # Select required columns
            df = dataframe[['timestamp', 'jetson_vdd_cpu_gpu_cv_mw', 'jetson_gpu_usage_percent',
                            'jetson_board_temperature_celsius', 'jetson_vdd_in_mw', 'jetson_cpu_usage_percent',
                            'jetson_ram_usage_mb', 'node_network_receive_bytes_total_KBps',
                            'node_network_transmit_bytes_total_KBps']]

            # Set 'timestamp' as the index
            df.set_index('timestamp', inplace=True)

            # Replace zero values with 0.01
            df.replace(0, 0.01, inplace=True)

            # Count the number of rows where zero values were replaced
            num_rows_with_zero_replaced = len(df[(df == 0.01).any(axis=1)])
            print(f"Number of rows where zero values were replaced in {filename}: {num_rows_with_zero_replaced}")

            # Check if there are any remaining zero values
            remaining_zeros = df[(df == 0).any(axis=1)]
            print(f"Remaining rows with zero values in {filename}: {len(remaining_zeros)}")

            # Keep only rows with timestamps every 10 seconds
            df = df[df.index.second % 5 == 0]

            # Stop processing after the first file for now
            break  # Remove this line if you want to process all files

    print("First few rows of the processed DataFrame:")
    print(df.head())
    print("Column names:")
    print(df.columns)

    return df


# Preprocess Dataset Function
def preprocess_dataset(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Print start and end date
    print("start date of dataset is :", df.index.min())
    print("end date of dataset is :", df.index.max())
    # Train and Test Split
    split_index = int(len(df) * 0.8)
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]

    # Print the start and end dates for each split
    print("Train start date:", train.index.min())
    print("Train end date:", train.index.max())
    print("Test start date:", test.index.min())
    print("Test end date:", test.index.max())
    print("Train set shape:", train.shape)
    print("Test set shape:", test.shape)

    seq_size = 40  # Number of time steps to look back

    # larger sequence size (look further back) may improve forecasting

    def to_sequence(x, y, seq_size=1):
        x_values = []
        y_values = []

        for i in range(len(x) - seq_size):
            x_values.append(x.iloc[i:(i + seq_size)].values)
            y_values.append(y.iloc[i:(i + seq_size)].values)  # Adjust this line for correct target shape

        return np.array(x_values), np.array(y_values)

    X_train, y_train = to_sequence(
        train[['jetson_vdd_cpu_gpu_cv_mw', 'jetson_vdd_cpu_gpu_cv_mw', 'jetson_board_temperature_celsius',
               'jetson_vdd_in_mw', 'jetson_cpu_usage_percent', 'jetson_ram_usage_mb',
               'node_network_receive_bytes_total_KBps', 'node_network_transmit_bytes_total_KBps']],
        train[['jetson_vdd_cpu_gpu_cv_mw', 'jetson_vdd_cpu_gpu_cv_mw', 'jetson_board_temperature_celsius',
               'jetson_vdd_in_mw', 'jetson_cpu_usage_percent', 'jetson_ram_usage_mb',
               'node_network_receive_bytes_total_KBps', 'node_network_transmit_bytes_total_KBps']],
        seq_size
    )

    X_test, y_test = to_sequence(
        test[['jetson_vdd_cpu_gpu_cv_mw', 'jetson_vdd_cpu_gpu_cv_mw', 'jetson_board_temperature_celsius',
              'jetson_vdd_in_mw', 'jetson_cpu_usage_percent', 'jetson_ram_usage_mb',
              'node_network_receive_bytes_total_KBps', 'node_network_transmit_bytes_total_KBps']],
        test[['jetson_vdd_cpu_gpu_cv_mw', 'jetson_vdd_cpu_gpu_cv_mw', 'jetson_board_temperature_celsius',
              'jetson_vdd_in_mw', 'jetson_cpu_usage_percent', 'jetson_ram_usage_mb',
              'node_network_receive_bytes_total_KBps', 'node_network_transmit_bytes_total_KBps']],
        seq_size
    )

    print("train X shape", X_train.shape)
    print("train Y shape", y_train.shape)
    print("test X shape", X_test.shape)
    print("test Y shape", y_test.shape)

    return X_train, y_train, X_test, y_test


class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r= self.model.fit(self.X_train, self.y_train, epochs=10, batch_size=128, validation_split=0.2, verbose=1)
        hist = r.history
        print("Fit history : ", hist)
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        eval_loss, eval_mape = self.model.evaluate(X_test, y_test)

        temp_loss.append(eval_loss)
        temp_mape.append(eval_mape)

        print(f"Eval Loss: {eval_loss} || Eval MAPE: {eval_mape}")
        return eval_loss, len(X_test), {"mape": eval_mape}

# Main Function
if __name__ == "__main__":
    # Load Dataset
    df = load_dataset()
    # Preprocess/split Dataset
    X_train, y_train, X_test, y_test = preprocess_dataset(df)

    #LSTM Model with Tensorflow GPU working
    model = Sequential()
    model.add( LSTM(128, activation='tanh', recurrent_activation='sigmoid', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
    model.add(LSTM(32, activation='tanh', recurrent_activation='sigmoid', return_sequences=False))
    model.add(RepeatVector(X_train.shape[1]))
    model.add(LSTM(32, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
    model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
    model.add(LSTM(128, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
    model.add(TimeDistributed(Dense(X_train.shape[2])))
    model.compile(optimizer='adam', loss='mae', metrics=["mape"])

    # Create Flower Client
    flower_client = FlowerClient()
    flower_client.X_train = X_train
    flower_client.y_train = y_train
    flower_client.X_test = X_test
    flower_client.y_test = y_test
    flower_client.model = model

    # Start Client
    fl.client.start_numpy_client(server_address=SERVER_ADDR, client=flower_client)

    # Save the trained model
    joblib.dump(model, 'trained_model_LSTM.joblib')

################ CALCULATING THE MAE AND MAPE FOR TRAIN AND TEST FOR THRESHOLDING ###################

    # Calculate MAE for training prediction
    trainPredict = model.predict(X_train)
    trainMAE = np.mean(np.abs(trainPredict - X_train), axis=1)
    # Print the mean of test MAE
    print("Mean of Train MAE:", np.mean(trainMAE))

    # # Plot
    # plt.figure(figsize=(8, 6))
    # plt.hist(trainMAE, bins=30)
    # plt.xlabel('Mean Absolute Error (MAE)')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Mean Absolute Error (MAE) in Training Prediction')
    # plt.savefig('train_mae_histogram.png')
    # plt.close()

    # Calculate MAPE for each sample
    trainActual = X_train
    trainMAPE = np.mean(np.abs(trainPredict - trainActual) / trainActual, axis=1) * 100
    # Print the mean of MAPE
    print("Mean of Train MAPE:", np.mean(trainMAPE))

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

    # Individual histograms for MAPE with feature names
    plt.figure(figsize=(15, 10))
    for i in range(trainMAPE.shape[1]):
        plt.subplot(4, 2, i + 1)  # Create a grid layout (4 rows, 2 columns)
        plt.hist(trainMAPE[:, i], bins=30, alpha=0.7, color='green')
        plt.title(f'MAPE for {feature_names[i]}')
        plt.xlabel('Mean Absolute Percentage Error (MAPE)')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('train_mape_histogram.png')
    plt.close()

    thresholds = []
    percentile = 95  # Set the desired percentile (e.g., 95th)
    for i in range(trainMAPE.shape[1]):
        threshold = np.percentile(trainMAPE[:, i], percentile)
        thresholds.append(threshold)
        print(f"Training Feature: {feature_names[i]}, Threshold ({percentile} Percentile): {threshold:.2f}")

    # # Plot
    # plt.figure(figsize=(8, 6))
    # plt.hist(trainMAPE, bins=30)
    # plt.xlabel('Mean Absolute Percentage Error (MAPE)')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Mean Absolute Percentage Error (MAPE) in Training Prediction')
    # plt.savefig('train_mape_histogram.png')
    # plt.close()

    # Calculate reconstruction loss (MAE) for testing dataset
    testPredict = model.predict(X_test)
    testMAE = np.mean(np.abs(testPredict - X_test), axis=1)

    # Print the mean of test MAE
    print("Mean of Test MAE:", np.mean(testMAE))

    # # Plot histogram
    # plt.figure(figsize=(8, 6))
    # plt.hist(testMAE, bins=30)
    # plt.xlabel('Test MAE')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Mean Absolute Error (MAE) in Test Prediction')
    # plt.savefig('test_mae_histogram.png')
    # plt.close()

    # Calculate MAPE for each sample
    testActual = X_test  # Assuming trainX contains the actual values
    testMAPE = np.mean(np.abs(testPredict - testActual) / testActual, axis=1) * 100

    # Print the mean of MAPE
    print("Mean of Test MAPE:", np.mean(testMAPE))

    # Individual histograms for MAPE with feature names
    plt.figure(figsize=(15, 10))
    for i in range(testMAPE.shape[1]):
        plt.subplot(4, 2, i + 1)  # Create a grid layout (4 rows, 2 columns)
        plt.hist(testMAPE[:, i], bins=30, alpha=0.7, color='green')
        plt.title(f'MAPE for {feature_names[i]}')
        plt.xlabel('Mean Absolute Percentage Error (MAPE)')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('test_mape_histogram.png')
    plt.close()

    # Percentile Method to define anomaly thresholds
    thresholds = []
    percentile = 95

    for i in range(testMAPE.shape[1]):
        threshold = np.percentile(testMAPE[:, i], percentile)
        thresholds.append(threshold)
        print(f"Test Feature: {feature_names[i]}, Threshold ({percentile} Percentile): {threshold:.2f}")

    # # Plot histogram of MAPE
    # plt.figure(figsize=(8, 6))
    # plt.hist(testMAPE, bins=30)
    # plt.xlabel('Mean Absolute Percentage Error (MAPE)')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Mean Absolute Percentage Error (MAPE) in Test Prediction')
    # plt.savefig('test_mape_histogram.png')
    # plt.close()

####################################################################################################
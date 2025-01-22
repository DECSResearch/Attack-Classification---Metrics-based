# Muhammad Hamza Karim

import os
import joblib
import random
import warnings
import argparse
import flwr as fl
import numpy as np
import keras as ke
import pandas as pd
import seaborn as sns
import tensorflow as tf
from typing import Tuple
from keras import Sequential
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Bidirectional
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


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

# Define RMSE function
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

#temp_loss = []
#temp_mape = []

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

def load_dataset():
    folder_path = os.path.join('.', 'Train_data', FOLDER_LOC)

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            dataframe = pd.read_csv(file_path)

            # Convert 'timestamp' column to datetime
            dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'])

            # Select required columns
            df = dataframe[['timestamp', 'jetson_gpu_usage_percent', 'jetson_board_temperature_celsius',
                            'jetson_cpu_usage_percent', 'jetson_ram_usage_mb']]

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

    # Scaling the dataset
    df_to_scale = df[['jetson_gpu_usage_percent', 'jetson_board_temperature_celsius', 'jetson_cpu_usage_percent',
                      'jetson_ram_usage_mb']]

    scaler_standard = StandardScaler()

    df_standard_scaled = pd.DataFrame(scaler_standard.fit_transform(df_to_scale), columns=df_to_scale.columns,
                                      index=df.index)


    train = df_standard_scaled  # Use the entire dataset as training data

    # Print the start and end dates for the dataset
    print("Train start date:", train.index.min())
    print("Train end date:", train.index.max())
    print("Train set shape:", train.shape)

    # Sequencing
    seq_size = 20  # Number of time steps to look back

    # larger sequence size (look further back) may improve forecasting

    def to_sequence(x, y, seq_size=1):
        x_values = []
        y_values = []

        for i in range(len(x) - seq_size):
            x_values.append(x.iloc[i:(i + seq_size)].values)
            y_values.append(y.iloc[i + seq_size])

        return np.array(x_values), np.array(y_values)

    X_train, y_train = to_sequence(train[['jetson_gpu_usage_percent', 'jetson_board_temperature_celsius',
                                        'jetson_cpu_usage_percent', 'jetson_ram_usage_mb']],
                                   train[['jetson_gpu_usage_percent', 'jetson_board_temperature_celsius',
                                      'jetson_cpu_usage_percent', 'jetson_ram_usage_mb']], seq_size)

    print("train X shape", X_train.shape)
    print("train Y shape", y_train.shape)

    return X_train, y_train


class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.X_train = None
        self.y_train = None
   #     self.X_test = None
   #     self.y_test = None
        self.model = None

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        r= self.model.fit(self.X_train, self.X_train, epochs=10, batch_size=128, validation_split=0.2, verbose=1)
        hist = r.history
        print("Fit history : ", hist)
        return self.model.get_weights(), len(self.X_train), {}

# We don't need evaluation as we do not have test data
#     def evaluate(self, parameters, config):
#         model.set_weights(parameters)
#         eval_loss, eval_mape = self.model.evaluate(X_train, y_train)
#
#         temp_loss.append(eval_loss)
#         temp_mape.append(eval_mape)
#
#         print(f"Eval Loss: {eval_loss} || Eval RMSE: {eval_rmse}")
#         return eval_loss, len(X_test), {"rmse": eval_rmse}

# Main Function
if __name__ == "__main__":
    # Load Dataset
    df = load_dataset()
    # Preprocess/split Dataset
   # X_train, y_train, X_test, y_test = preprocess_dataset(df)
    X_train, y_train = preprocess_dataset(df)

    # Define the model
    # LSTM
    model = Sequential()
    model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(32, activation='tanh', recurrent_activation='sigmoid', return_sequences=False))
    model.add(RepeatVector(X_train.shape[1]))
    model.add(LSTM(32, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
    model.add(LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
    model.add(TimeDistributed(Dense(X_train.shape[2])))

    # # BiLSTM
    # model = Sequential()
    # model.add(Bidirectional(LSTM(64, activation='tanh', recurrent_activation='sigmoid', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True)))
    # # model.add(Bidirectional(LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True)))
    # model.add(Bidirectional(LSTM(32, activation='tanh', recurrent_activation='sigmoid', return_sequences=False)))
    # model.add(RepeatVector(trainX.shape[1]))
    # # model.add(Bidirectional(LSTM(32, activation='tanh', recurrent_activation='sigmoid', return_sequences=True)))
    # model.add(Bidirectional(LSTM(32, activation='tanh', recurrent_activation='sigmoid', return_sequences=True)))
    # model.add(Bidirectional(LSTM(64, activation='tanh', recurrent_activation='sigmoid', return_sequences=True)))
    # model.add(TimeDistributed(Dense(trainX.shape[2])))

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=[rmse])

    # Create Flower Client
    flower_client = FlowerClient()
    flower_client.X_train = X_train
    flower_client.y_train = y_train
 #   flower_client.X_test = X_test
 #   flower_client.y_test = y_test
    flower_client.model = model

    # Start Client
    fl.client.start_numpy_client(server_address=SERVER_ADDR, client=flower_client)

    # Save the trained model
    joblib.dump(model, 'NFL-LSTM.joblib')

################ CALCULATING THE loss AND RMSE FOR TRAIN AND TEST FOR THRESHOLDING ###################

    # Calculate MAE MSE & RMSE for training prediction
    trainPredict = model.predict(X_train)

    trainMAE = np.mean(np.abs(trainPredict - X_train), axis=1)
    # Print the mean of test MAE
    print("Mean of Train MAE:", np.mean(trainMAE))
    print(trainMAE.shape)

    # Calculate MSE for training predictions
    trainMSE = np.mean(np.square(trainPredict - X_train), axis=1)

    # Calculate RMSE for training predictions
    trainRMSE = np.sqrt(trainMSE)

    # Print the mean of Train MSE and Train RMSE
    print("Mean of Train MSE:", np.mean(trainMSE))
    print("Mean of Train RMSE:", np.mean(trainRMSE))
    print("Shape of trainMSE:", trainMSE.shape)
    print("Shape of trainRMSE:", trainRMSE.shape)

    trainRMSE_mean = np.mean(trainRMSE, axis=1)

    # Print the shape and mean of trainRMSE_mean
    print("Shape of trainRMSE_mean:", trainRMSE_mean.shape)
    print("Mean of trainRMSE_mean:", np.mean(trainRMSE_mean))

    # Compute the 95th and 99.5th percentiles
    threshold_95 = np.percentile(trainRMSE_mean, 95)
    threshold_99 = np.percentile(trainRMSE_mean, 99.5)
    print("95th Percentile Threshold:", threshold_95)
    print("99.5th Percentile Threshold:", threshold_99)

    # Plot the distribution of the mean RMSE values
    plt.figure(figsize=(12, 6))
    sns.histplot(trainRMSE_mean, bins=50, color='royalblue', kde=True, label='Mean RMSE')
    plt.axvline(threshold_99, color='red', linestyle='--', linewidth=2, label='Threshold')
    # Annotate the threshold value on the plot
    plt.text(threshold_99, plt.gca().get_ylim()[1] * 0.9, f'{threshold_99:.3f}',
             color='red', fontsize=12, fontweight='bold', ha='left', va='bottom')
    x_ticks = np.arange(0, 1.1, 0.1)
    plt.xticks(x_ticks, fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Distribution of Mean Train RMSE', fontsize=18, fontweight='bold')
    plt.xlabel('Mean RMSE', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12, loc='upper right', frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig('Distribution of Mean Train RMSE LSTM.png')
    plt.close()
    # plt.show()

####################################################################################################
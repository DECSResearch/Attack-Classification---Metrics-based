# Muhammad Hamza Karim

import os
import joblib
import keras
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Limit TensorFlow memory usage to 1 GB
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    except RuntimeError as e:
        print(e)

# Function to convert a dataframe into sequences
# Sequencing
seq_size = 20  # Number of time steps to look back
def to_sequence(x, y, seq_size=1):
    x_values = []
    y_values = []

    for i in range(len(x) - seq_size):
        x_values.append(x.iloc[i:(i + seq_size)].values)
        y_values.append(y.iloc[i + seq_size])

    return np.array(x_values), np.array(y_values)

# Define RMSE function
@keras.utils.register_keras_serializable()
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

# Load the trained model
model = joblib.load('NFL-LSTM.joblib')

# Load the dataset
data1 = pd.read_csv('Test_data/Nano07V2_gt.csv')

# Select only the 'timestamp', 'jetson_gpu_usage_percent', 'jetson_cpu_usage_percent', and 'ground_truth' columns
combined_data = data1[['timestamp', 'jetson_gpu_usage_percent', 'jetson_cpu_usage_percent', 'jetson_board_temperature_celsius', 'jetson_ram_usage_mb', 'ground_truth']]

# Convert 'timestamp' to datetime format
combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'])

# Replace zero values with a small value (0.01) for all columns except 'ground_truth'
columns_to_replace = combined_data.columns.difference(['ground_truth'])
combined_data[columns_to_replace] = combined_data[columns_to_replace].replace(0, 0.01)

# Verify the changes
print(combined_data.head())

# Filter rows where the timestamp's second is divisible by 5
# combined_data = combined_data[combined_data['timestamp'].dt.second % 5 == 0]

# Extract the columns to scale
df_to_scale = combined_data[['jetson_gpu_usage_percent', 'jetson_cpu_usage_percent', 'jetson_board_temperature_celsius', 'jetson_ram_usage_mb']]

# Initialize the StandardScaler
scaler_standard = StandardScaler()

# Perform scaling
df_standard_scaled_1 = pd.DataFrame(scaler_standard.fit_transform(df_to_scale),
                                  columns=df_to_scale.columns,
                                  index=combined_data.index)

# Add the 'timestamp' column back to the scaled data
df_standard_scaled_1['timestamp'] = combined_data['timestamp']

# Convert the combined dataset into sequences
combined_X, combined_Y = to_sequence(df_standard_scaled_1[['jetson_gpu_usage_percent', 'jetson_cpu_usage_percent', 'jetson_board_temperature_celsius', 'jetson_ram_usage_mb']], df_standard_scaled_1[['jetson_gpu_usage_percent', 'jetson_cpu_usage_percent', 'jetson_board_temperature_celsius', 'jetson_ram_usage_mb']], seq_size)

print("combined X shape", combined_X.shape)
print("combined Y shape", combined_Y.shape)

# Measure the time before starting the prediction
start_time = time.time()

# Use the trained model to predict the reconstruction errors (MAPE) on the combined dataset
combined_predict = model.predict(combined_X)

# Measure the time after prediction
end_time = time.time()

# Calculate total inference time
inference_time = end_time - start_time
print(f"Total inference time: {inference_time:.2f} seconds")

# Calculate combined MAPE
# combined_mape = np.mean(np.abs(combined_predict - combined_X) / combined_X, axis=1) * 100
combined_mae = np.mean(np.abs(combined_predict - combined_X), axis=1)
combined_mae_combined = np.mean(combined_mae, axis=1)

combined_mse = np.mean(np.square(combined_predict - combined_X), axis=1)
combined_mse_combined = np.mean(combined_mse, axis=1)

combined_rmse = np.sqrt(combined_mse)
combined_rmse_combined = np.mean(combined_rmse, axis=1)

# Thresholding using MAPE
# max_trainRMSE = threshold_99
max_trainRMSE = 0.65

# Capture all details in a DataFrame for easy plotting
anomaly_df = pd.DataFrame(df_standard_scaled_1[seq_size:])
anomaly_df['combinedRMSE'] = combined_rmse_combined
anomaly_df['max_trainRMSE'] = max_trainRMSE
anomaly_df['anomaly'] = anomaly_df['combinedRMSE'] > max_trainRMSE
anomaly_df[['jetson_gpu_usage_percent', 'jetson_cpu_usage_percent', 'jetson_board_temperature_celsius', 'jetson_ram_usage_mb']] = df_standard_scaled_1[seq_size:][['jetson_gpu_usage_percent', 'jetson_cpu_usage_percent', 'jetson_board_temperature_celsius', 'jetson_ram_usage_mb']]

# Add the anomaly flag to the original test dataset
combined_data['anomaly'] = False
combined_data.loc[anomaly_df.index, 'anomaly'] = anomaly_df['anomaly'].values

# Separate normal and anomalous data for plotting
normal_data = combined_data[~combined_data['anomaly']]
anomalous_data = combined_data[combined_data['anomaly']]

# Plot combined RMSE with adjusted highlighted regions
plt.figure(figsize=(12, 8))

# Highlight the area within the threshold in green
plt.fill_between(anomaly_df.index, anomaly_df['max_trainRMSE'], anomaly_df['combinedRMSE'],
                 where=anomaly_df['combinedRMSE'] <= anomaly_df['max_trainRMSE'],
                 color='green', alpha=0.2, label='Normal Region')

# Highlight the area above the threshold in red
plt.fill_between(anomaly_df.index, anomaly_df['max_trainRMSE'], anomaly_df['combinedRMSE'],
                 where=anomaly_df['combinedRMSE'] > anomaly_df['max_trainRMSE'],
                 color='red', alpha=0.2, label='Anomalous Region')

# Plot the RMSE values
sns.lineplot(x=anomaly_df.index, y=anomaly_df['combinedRMSE'], label='Combined RMSE', color='blue', linewidth=2)

# Plot the threshold line
sns.lineplot(x=anomaly_df.index, y=anomaly_df['max_trainRMSE'], label='Threshold', color='red', linestyle='--', linewidth=2)

# Annotate the threshold value
threshold_value = anomaly_df['max_trainRMSE'].iloc[0]  # Assuming the threshold is constant
plt.text(x=anomaly_df.index[-1],
         y=threshold_value,
         s=f'Threshold: {threshold_value:.2f}',
         color='red',
         fontsize=12,
         verticalalignment='bottom',
         horizontalalignment='right',
         bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))

# Customize the plot
plt.title('Anomaly Detection on Testing Data (Version 1)', fontsize=16, fontweight='bold')
plt.xlabel('Index', fontsize=14)
plt.ylabel('Root Mean Squared Error (RMSE)', fontsize=14)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Add background color to make the plot more appealing
plt.gcf().set_facecolor('white')
plt.tight_layout()
plt.savefig('Testing-V2-FL-LSTM-FLWR.png')
plt.close()


# Compare ground truth with predicted anomalies (model's 'anomaly' column)
y_true = combined_data['ground_truth']  # Actual labels (0 or 1)
y_pred = combined_data['anomaly'].astype(int)  # Predicted anomalies (0 or 1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print(f"Confusion Matrix:\n{cm}")

# Precision, Recall, F1-Score, and Accuracy
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

# Print the metrics
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")

# Plot the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('CM-TestingV2-FL-LSTM-FLWR.png')
plt.close()

import time
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from logging import getLogger

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class GPUUsageScaler(BaseEstimator, TransformerMixin):
    """Custom scaler for GPU usage data that handles idle states"""
    def __init__(self, idle_threshold=1.0, idle_scale=-5.0):
        self.mean_idle_ = None
        self.std_idle_ = None
        self.mean_active_ = None
        self.std_active_ = None
        self.idle_threshold = idle_threshold
        self.idle_scale = idle_scale

    def fit(self, X, y=None):
        X = np.asarray(X).reshape(-1, 1)
        
        idle_mask = X <= self.idle_threshold
        active_mask = X > self.idle_threshold
        
        if np.any(idle_mask):
            self.mean_idle_ = np.mean(X[idle_mask])
            self.std_idle_ = np.std(X[idle_mask]) if np.std(X[idle_mask]) > 0 else 0.1
        else:
            self.mean_idle_ = 0
            self.std_idle_ = 0.1
        
        if np.any(active_mask):
            self.mean_active_ = np.mean(X[active_mask])
            self.std_active_ = np.std(X[active_mask]) if np.std(X[active_mask]) > 0 else 1
        else:
            self.mean_active_ = self.idle_threshold
            self.std_active_ = 1
            
        return self

    def transform(self, X):
        X = np.asarray(X).reshape(-1, 1)
        scaled = np.zeros_like(X)
        
        idle_mask = X <= self.idle_threshold
        active_mask = X > self.idle_threshold
        
        if np.any(idle_mask):
            scaled[idle_mask] = self.idle_scale
            
        if np.any(active_mask):
            scaled[active_mask] = (X[active_mask] - self.mean_active_) / self.std_active_
            
        return scaled

    def inverse_transform(self, X):
        X = np.asarray(X).reshape(-1, 1)
        original = np.zeros_like(X)
        
        idle_mask = X <= (self.idle_scale / 2)
        active_mask = X > (self.idle_scale / 2)
        
        if np.any(idle_mask):
            original[idle_mask] = 0.0
            
        if np.any(active_mask):
            original[active_mask] = (X[active_mask] * self.std_active_) + self.mean_active_
            
        original = np.clip(original, 0, 100)
        
        return original

def to_sequence(x, y, seq_size=1):
    x_values = []
    y_values = []
    
    for i in range(len(x)-seq_size):
        x_values.append(x.iloc[i:(i+seq_size)].values)
        y_values.append(y.iloc[i+seq_size])
    
    return np.array(x_values), np.array(y_values)

def process_dataset(file_path, model, seq_size=20):
    # Load and preprocess the data
    data = pd.read_csv(file_path)
    
    combined_data = data[['timestamp', 'jetson_gpu_usage_percent', 'jetson_cpu_usage_percent', 
                         'jetson_board_temperature_celsius', 'jetson_ram_usage_mb', 'ground_truth']]
    
    # Convert timestamp to datetime
    combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'])
    
    # Replace zero values with 0.01
    columns_to_replace = combined_data.columns.difference(['ground_truth'])
    combined_data[columns_to_replace] = combined_data[columns_to_replace].replace(0, 0.01)
    
    # Scale the data
    df_to_scale = combined_data[['jetson_gpu_usage_percent', 'jetson_cpu_usage_percent', 
                                'jetson_board_temperature_celsius', 'jetson_ram_usage_mb']]
    
    # Initialize scalers
    gpu_scaler = GPUUsageScaler(idle_threshold=1.0, idle_scale=-5.0)
    cpu_scaler = StandardScaler()
    temp_scaler = StandardScaler()
    ram_scaler = StandardScaler()

    # Scale each feature
    scaled_gpu = gpu_scaler.fit_transform(df_to_scale['jetson_gpu_usage_percent'].values.reshape(-1, 1))
    scaled_cpu = cpu_scaler.fit_transform(df_to_scale['jetson_cpu_usage_percent'].values.reshape(-1, 1))
    scaled_temp = temp_scaler.fit_transform(df_to_scale['jetson_board_temperature_celsius'].values.reshape(-1, 1))
    scaled_ram = ram_scaler.fit_transform(df_to_scale['jetson_ram_usage_mb'].values.reshape(-1, 1))

    # Create scaled DataFrame
    df_scaled = pd.DataFrame({
        'jetson_gpu_usage_percent': scaled_gpu.flatten(),
        'jetson_cpu_usage_percent': scaled_cpu.flatten(),
        'jetson_board_temperature_celsius': scaled_temp.flatten(),
        'jetson_ram_usage_mb': scaled_ram.flatten()
    }, index=df_to_scale.index)
    
    # Create sequences
    combined_X, combined_Y = to_sequence(
        df_scaled[['jetson_gpu_usage_percent', 'jetson_cpu_usage_percent', 
                  'jetson_board_temperature_celsius', 'jetson_ram_usage_mb']], 
        df_scaled[['jetson_gpu_usage_percent', 'jetson_cpu_usage_percent', 
                  'jetson_board_temperature_celsius', 'jetson_ram_usage_mb']], 
        seq_size
    )
    
    # Make predictions and calculate metrics
    predictions = model.predict(combined_X)
    combined_rmse = np.sqrt(np.mean(np.square(predictions - combined_X), axis=(1, 2)))
    
    return combined_data, df_scaled, combined_X, combined_rmse

# Load models
models = {
    #'LSTM': tf.keras.models.load_model("results/new_custom_metrics/OpenFL_customscaler/lstm/output_model", compile=False),
    'BiLSTM': tf.keras.models.load_model("results/new_custom_metrics/OpenFL_customscaler/bilstm/output_model", compile=False),
}

# Define datasets
datasets = [
    ('V1', 'data/2025/testdata/Nano08V1_gt.csv'),
    ('V2', 'data/2025/testdata/Nano07V2_gt.csv'),
    ('V3', 'data/2025/testdata/Nano07V3_gt.csv'),
    ('V2_processed', 'data/2025/testdata/Nano07V2_gt_processed.csv'),
    ('V3_processed', 'data/2025/testdata/Nano07V3_gt_processed.csv'),    
]

# Process each model
for model_name, model in models.items():
    for dataset_name, dataset_path in datasets:
        # Process dataset
        combined_data, df_standard_scaled, combined_X, combined_rmse = process_dataset(dataset_path, model)
        
        threshold = 0.4824
        
        # 1. Plot RMSE Distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(combined_rmse, bins=50, color='royalblue', kde=True, label='Mean RMSE')
        plt.axvline(threshold, color='red', linestyle='--', linewidth=2, label='99.5th Percentile Threshold')
        plt.text(threshold, plt.gca().get_ylim()[1] * 0.9, f'{threshold:.3f}',
                 color='red', fontsize=12, fontweight='bold', ha='left', va='bottom')
        
        plt.title(f'{model_name} - {dataset_name} RMSE Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('RMSE', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{model_name}_{dataset_name}_rmse_dist.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Plot Anomaly Detection
        plt.figure(figsize=(15, 8))
        
        # Create anomaly DataFrame
        anomaly_df = pd.DataFrame(df_standard_scaled[20:])
        anomaly_df['combinedRMSE'] = combined_rmse
        anomaly_df['max_trainRMSE'] = threshold
        anomaly_df['anomaly'] = anomaly_df['combinedRMSE'] > threshold
        
        # Add anomaly flag to original dataset
        combined_data['anomaly'] = False
        combined_data.loc[anomaly_df.index, 'anomaly'] = anomaly_df['anomaly'].values
        
        plt.fill_between(anomaly_df.index, anomaly_df['max_trainRMSE'], anomaly_df['combinedRMSE'], 
                        where=anomaly_df['combinedRMSE'] <= anomaly_df['max_trainRMSE'], 
                        color='green', alpha=0.2, label='Normal Region')
        plt.fill_between(anomaly_df.index, anomaly_df['max_trainRMSE'], anomaly_df['combinedRMSE'], 
                        where=anomaly_df['combinedRMSE'] > anomaly_df['max_trainRMSE'], 
                        color='red', alpha=0.2, label='Anomalous Region')
        
        plt.plot(anomaly_df.index, anomaly_df['combinedRMSE'], label='Combined RMSE', color='blue', linewidth=2)
        plt.plot(anomaly_df.index, anomaly_df['max_trainRMSE'], label='Threshold', color='red', linestyle='--', linewidth=2)
        
        # Calculate metrics
        y_true = combined_data['ground_truth'][20:]
        y_pred = combined_data['anomaly'][20:].astype(int)
        
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        
        plt.title(f'{model_name} - {dataset_name} Anomaly Detection\nPrecision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, Accuracy: {accuracy:.2f}',
                 fontsize=14, pad=20)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Root Mean Squared Error (RMSE)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{model_name}_{dataset_name}_anomaly.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Plot Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Anomaly'], 
                   yticklabels=['Normal', 'Anomaly'])
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'{model_name}_{dataset_name}_CM.png', dpi=300, bbox_inches='tight')
        plt.close()
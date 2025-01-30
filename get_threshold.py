import numpy as np
import pandas as pd
from typing import Tuple, Dict
from logging import getLogger
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from keras.saving import register_keras_serializable

logger = getLogger(__name__)

@register_keras_serializable()
def rmse(y_true, y_pred):
    """Custom RMSE metric for model evaluation"""
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

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

def to_sequence(data, seq_size=1):
    """Convert DataFrame into sequences for LSTM"""
    x_values = []
    y_values = []
    
    for i in range(len(data) - seq_size):
        x_values.append(data.iloc[i:(i + seq_size)].values)
        y_values.append(data.iloc[i + seq_size].values)
        
    return np.array(x_values), np.array(y_values)

def load_and_preprocess_data(file_path: str, seq_size: int = 20) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Load and preprocess data from CSV file for LSTM processing"""
    # Load CSV file
    dataframe = pd.read_csv(file_path)
    dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'])
    
    # Filter relevant columns
    df = dataframe[['timestamp', 'jetson_gpu_usage_percent', 'jetson_board_temperature_celsius', 
                    'jetson_cpu_usage_percent', 'jetson_ram_usage_mb']]
    df.set_index('timestamp', inplace=True)
    
    # Count zero values
    num_rows_with_zero_replaced = len(df[(df == 0.01).any(axis=1)])
    remaining_zeros = df[(df == 0).any(axis=1)]
    logger.info(f"Number of rows where zero values were replaced: {num_rows_with_zero_replaced}")
    logger.info(f"Remaining rows with zero values: {len(remaining_zeros)}")

    # Initialize scalers
    scalers = {
        'gpu': GPUUsageScaler(idle_threshold=1.0, idle_scale=-5.0),
        'temp': StandardScaler(),
        'cpu': StandardScaler(),
        'ram': StandardScaler()
    }

    # Scale features individually
    scaled_features = {
        'jetson_gpu_usage_percent': scalers['gpu'].fit_transform(df['jetson_gpu_usage_percent'].values.reshape(-1, 1)),
        'jetson_board_temperature_celsius': scalers['temp'].fit_transform(df['jetson_board_temperature_celsius'].values.reshape(-1, 1)),
        'jetson_cpu_usage_percent': scalers['cpu'].fit_transform(df['jetson_cpu_usage_percent'].values.reshape(-1, 1)),
        'jetson_ram_usage_mb': scalers['ram'].fit_transform(df['jetson_ram_usage_mb'].values.reshape(-1, 1))
    }

    # Create scaled DataFrame
    df_scaled = pd.DataFrame({
        col: scaled.flatten() for col, scaled in scaled_features.items()
    }, index=df.index)
    
    # Create sequences
    X_train, y_train = to_sequence(df_scaled, seq_size)
    
    # Log data information
    logger.info(f'Data > X Shape : {X_train.shape}')
    logger.info(f'Data period: {df.index.min()} to {df.index.max()}')
    
    return X_train, y_train, scalers

def load_data_shard(shard_num: int, collaborator_count: int, categorical: bool = True,
                    channels_last: bool = True, **kwargs) -> Tuple[tuple, np.ndarray]:
    """Load a shard of the dataset for distributed training"""
    seq_size = kwargs.get('seq_size', 20)
    file_path = kwargs.get('file_path', '../data/2025/nano07_short.csv')
    
    X_train, y_train, _ = load_and_preprocess_data(file_path, seq_size)
    
    input_shape = (seq_size, 4)  # 4 features, seq_size timesteps
    
    return input_shape, X_train, y_train

def calculate_metrics(model, X_train: np.ndarray, save_dir: str = "results", client_name: str = "client"):
    """Calculate and visualize model performance metrics"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate predictions
    train_predict = model.predict(X_train)
    
    # Calculate MSE and RMSE
    train_mse = np.mean(np.square(train_predict - X_train), axis=1)
    train_rmse = np.sqrt(train_mse)
    train_rmse_mean = np.mean(train_rmse, axis=1)
    train_mse_mean = np.mean(train_mse, axis=1)

    # Calculate thresholds
    threshold_50 = np.percentile(train_rmse_mean, 50)
    threshold_95 = np.percentile(train_rmse_mean, 95)
    threshold_99 = np.percentile(train_rmse_mean, 99.5)

    # Log metrics
    metrics = {
        'mean_mse': float(np.mean(train_mse_mean)),
        'max_mse': float(np.max(train_mse_mean)),
        'mean_rmse': float(np.mean(train_rmse_mean)),
        'max_rmse': float(np.max(train_rmse_mean)),
        'threshold_50': float(threshold_50),
        'threshold_95': float(threshold_95),
        'threshold_99': float(threshold_99)
    }
    
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value}")

    # Plot RMSE distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(train_rmse_mean, bins=50, color='royalblue', kde=True, label='Mean RMSE')
    plt.axvline(threshold_99, color='red', linestyle='--', linewidth=2, label='99.5th Percentile')
    plt.text(threshold_99, plt.gca().get_ylim()[1] * 0.9, 
             f'{threshold_99:.3f}', color='red', fontsize=12, 
             fontweight='bold', ha='left', va='bottom')
    
    plt.title(f'RMSE Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('RMSE', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_dir / f'rmse_dist.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot predictions vs actual values
    plt.figure(figsize=(15, 6))
    plt.plot(X_train.flatten()[:200], label='Actual', color='blue', alpha=0.7)
    plt.plot(train_predict.flatten()[:200], label='Predicted', color='red', alpha=0.7)
    plt.title(f'{client_name} - Actual vs Predicted Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Scaled Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / f'{client_name}_predictions.png')
    plt.close()

    return metrics

def Calculate(df: pd.DataFrame, model, client_name: str = "client", save_dir: str = "results"):
    """Calculate and visualize model performance for a given DataFrame"""
    # Convert timestamp and set as index if not already done
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
    # Select relevant columns
    columns = [
        'jetson_gpu_usage_percent',
        'jetson_board_temperature_celsius',
        'jetson_cpu_usage_percent',
        'jetson_ram_usage_mb'
    ]
    df = df[columns]
    
    # Initialize and fit scalers
    gpu_scaler = GPUUsageScaler(idle_threshold=1.0, idle_scale=-5.0)
    temp_scaler = StandardScaler()
    cpu_scaler = StandardScaler()
    ram_scaler = StandardScaler()
    
    # Scale features
    scaled_gpu = gpu_scaler.fit_transform(df['jetson_gpu_usage_percent'].values.reshape(-1, 1))
    scaled_temp = temp_scaler.fit_transform(df['jetson_board_temperature_celsius'].values.reshape(-1, 1))
    scaled_cpu = cpu_scaler.fit_transform(df['jetson_cpu_usage_percent'].values.reshape(-1, 1))
    scaled_ram = ram_scaler.fit_transform(df['jetson_ram_usage_mb'].values.reshape(-1, 1))
    
    # Create scaled DataFrame
    df_scaled = pd.DataFrame({
        'jetson_gpu_usage_percent': scaled_gpu.flatten(),
        'jetson_board_temperature_celsius': scaled_temp.flatten(),
        'jetson_cpu_usage_percent': scaled_cpu.flatten(),
        'jetson_ram_usage_mb': scaled_ram.flatten()
    }, index=df.index)
    
    # Create sequences
    x_seq = []
    for i in range(len(df_scaled) - 20):
        x_seq.append(df_scaled.iloc[i:(i + 20)].values)
    X_train = np.array(x_seq)
    
    # Calculate metrics and create visualizations
    metrics = calculate_metrics(model, X_train, save_dir, client_name)
    return metrics

def evaluate_model(model_path: str, data_paths: list, save_dir: str = "evaluation_results"):
    """Evaluate model on multiple datasets"""
    # Load model with custom objects
    custom_objects = {'rmse': rmse}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    
    results = {}
    for data_path in data_paths:
        client_name = Path(data_path).stem
        logger.info(f"Processing {client_name}")
        
        # Load and process data
        df = pd.read_csv(data_path)
        metrics = Calculate(df, model, client_name, save_dir)
        results[client_name] = metrics
        
    return results

if __name__ == "__main__":
    # Example usage
    model_path = "results/new_custom_metrics/OpenFL_customscaler/lstm/output_model"
    #model_path = "results/new_custom_metrics/OpenFL_customscaler/bilstm/output_model"
    data_paths = [
        'data/2025/nano07_short.csv'
        #'data/2025/nano08_short.csv'
    ]
    
    results = evaluate_model(
        model_path=model_path,
        data_paths=data_paths,
        save_dir="evaluation_results"
    )
    
    print("\nEvaluation Results:")
    for client, metrics in results.items():
        print(f"\nClient: {client}")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
import numpy as np
import pandas as pd
from typing import Tuple
from logging import getLogger
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

logger = getLogger(__name__)
    
class GPUUsageScaler(BaseEstimator, TransformerMixin):
    """
    A custom scaler for GPU usage data that handles idle and active states differently.
    Inherits from sklearn's BaseEstimator and TransformerMixin.
    
    Parameters:
    -----------
    idle_threshold : float, default=1.0
        Values below or equal to this threshold are considered idle state
    idle_scale : float, default=-5.0
        The value to which idle state data will be scaled
    """
    def __init__(self, idle_threshold=1.0, idle_scale=-5.0):
        self.mean_idle_ = None       # Mean of idle state values
        self.std_idle_ = None        # Standard deviation of idle state values
        self.mean_active_ = None     # Mean of active state values
        self.std_active_ = None      # Standard deviation of active state values
        self.idle_threshold = idle_threshold
        self.idle_scale = idle_scale

    def fit(self, X, y=None):
        """
        Fit the scaler to the data by computing necessary statistics.
        
        Parameters:
        -----------
        X : array-like
            The data to fit
        y : None
            Ignored. Exists for compatibility with sklearn API.
            
        Returns:
        --------
        self : object
            Returns the instance itself
        """
        X = np.asarray(X).reshape(-1, 1)
        
        # Split data into idle and active states
        idle_mask = X <= self.idle_threshold
        active_mask = X > self.idle_threshold
        
        # Compute statistics for idle state
        if np.any(idle_mask):
            self.mean_idle_ = np.mean(X[idle_mask])
            self.std_idle_ = np.std(X[idle_mask]) if np.std(X[idle_mask]) > 0 else 0.1
        else:
            self.mean_idle_ = 0
            self.std_idle_ = 0.1
        
        # Compute statistics for active state
        if np.any(active_mask):
            self.mean_active_ = np.mean(X[active_mask])
            self.std_active_ = np.std(X[active_mask]) if np.std(X[active_mask]) > 0 else 1
        else:
            self.mean_active_ = self.idle_threshold
            self.std_active_ = 1
            
        return self

    def transform(self, X):
        """
        Transform the data using the fitted scaler.
        
        Parameters:
        -----------
        X : array-like
            The data to transform
            
        Returns:
        --------
        scaled : ndarray
            The scaled data
        """
        X = np.asarray(X).reshape(-1, 1)
        scaled = np.zeros_like(X)
        
        # Apply different scaling for idle and active states
        idle_mask = X <= self.idle_threshold
        active_mask = X > self.idle_threshold
        
        if np.any(idle_mask):
            # Map all idle state values to idle_scale
            scaled[idle_mask] = self.idle_scale
            
        if np.any(active_mask):
            # Standardize active state values
            scaled[active_mask] = (X[active_mask] - self.mean_active_) / self.std_active_
            
        return scaled

    def inverse_transform(self, X):
        """
        Transform scaled data back to original scale.
        
        Parameters:
        -----------
        X : array-like
            The scaled data to inverse transform
            
        Returns:
        --------
        original : ndarray
            The inverse transformed data
        """
        X = np.asarray(X).reshape(-1, 1)
        original = np.zeros_like(X)
        
        # Determine states based on scaled values
        idle_mask = X <= (self.idle_scale / 2)
        active_mask = X > (self.idle_scale / 2)
        
        if np.any(idle_mask):
            # Map idle state back to zero
            original[idle_mask] = 0.0
            
        if np.any(active_mask):
            # Inverse transform active state values
            original[active_mask] = (X[active_mask] * self.std_active_) + self.mean_active_
            
        # Ensure values are within valid range
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

def load_and_preprocess_data(file_path: str, seq_size: int = 20) -> np.ndarray:
    """Load and preprocess data from CSV file for LSTM processing"""
    # Load CSV file
    dataframe = pd.read_csv(file_path)
    # Convert 'timestamp' column to datetime
    dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'])
    
    # Filter relevant columns
    df = dataframe[['timestamp', 'jetson_gpu_usage_percent', 'jetson_board_temperature_celsius', 
                    'jetson_cpu_usage_percent', 'jetson_ram_usage_mb']]
    df.set_index('timestamp', inplace=True)
    
    # Count the number of rows where zero values were replaced with 0.01
    num_rows_with_zero_replaced = len(df[(df == 0.01).any(axis=1)])
    print(f"Number of rows where zero values were replaced: {num_rows_with_zero_replaced}")
    
    remaining_zeros = df[(df == 0).any(axis=1)]
    print(f"Remaining rows with zero values: {len(remaining_zeros)}")

    # Separate scaling for each feature
    gpu_scaler = GPUUsageScaler(idle_threshold=1.0, idle_scale=-5.0)
    temp_scaler = StandardScaler()
    cpu_scaler = StandardScaler()
    ram_scaler = StandardScaler()

    # Scale features individually
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
    X_train, y_train = to_sequence(df_scaled, seq_size)
    
    # Log data shapes
    logger.info(f'Data > X Shape : {X_train.shape}')
    logger.info(f'Data period: {df.index.min()} to {df.index.max()}')
    
    return X_train, y_train

def load_data_shard(shard_num: int, collaborator_count: int, categorical: bool = True,
                    channels_last: bool = True, **kwargs) -> Tuple[tuple, np.ndarray]:
    """Load a shard of the dataset for distributed training."""
    seq_size = kwargs.get('seq_size', 20)
    file_path = kwargs.get('file_path', '../data/2025/nano07_short.csv')
    
    X_train, y_train = load_and_preprocess_data(file_path, seq_size)
    
    input_shape = (seq_size, 4)  # 4 features, seq_size timesteps
    
    return input_shape, X_train, y_train
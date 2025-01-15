import numpy as np
import pandas as pd
from typing import Tuple
from logging import getLogger
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = getLogger(__name__)

def to_sequence(x, y, seq_size=1):
    """
    Convert time series data into sequences for LSTM processing.
    
    Args:
        x: Input features DataFrame
        y: Target values DataFrame
        seq_size: Number of timesteps to include in each sequence
    """
    x_values = []
    y_values = []
    
    for i in range(len(x) - seq_size):
        x_values.append(x.iloc[i:(i + seq_size)].values)
        y_values.append(y.iloc[i:(i + seq_size)].values)
    
    return np.array(x_values), np.array(y_values)

def load_and_preprocess_data(file_path: str, seq_size: int = 20, train_size: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess data from CSV file for LSTM processing.
    
    Args:
        file_path: Path to the CSV file
        seq_size: Number of timesteps to use in sequences
        train_size: Proportion of data to use for training
        
    Returns:
        Tuple containing processed training and test data
    """
    # Load CSV file
    df = pd.read_csv(file_path)
        
    # Convert timestamp and set as index
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
    
    # Scaler
    df_to_scale = df[['jetson_gpu_usage_percent', 'jetson_board_temperature_celsius', 'jetson_cpu_usage_percent', 'jetson_ram_usage_mb']]
    scaler_standard = StandardScaler()
    df = pd.DataFrame(scaler_standard.fit_transform(df_to_scale), columns=df_to_scale.columns, index=df.index)
    
    # Replace zeros with small value to avoid division issues
    df.replace(0, 0.01, inplace=True)
        
    # Split into train and test sets
    split_index = int(len(df) * train_size)
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
        
    # Create sequences
    X_train, y_train = to_sequence(train, train, seq_size)
    X_test, y_test = to_sequence(test, test, seq_size)
        
    # Log data shapes
    logger.info(f'Data > X_train Shape : {X_train.shape}')
    logger.info(f'Data > X_test Shape : {X_test.shape}')
    logger.info(f'Train period: {train.index.min()} to {train.index.max()}')
    logger.info(f'Test period: {test.index.min()} to {test.index.max()}')
        
    return X_train, X_test

def load_data_shard(shard_num: int, collaborator_count: int, categorical: bool = True,
                   channels_last: bool = True, **kwargs) -> Tuple[tuple, np.ndarray, np.ndarray]:
    """
    Load a shard of the dataset for distributed training.
    
    Args:
        shard_num: Index of the current shard
        collaborator_count: Total number of collaborators
        categorical: Whether to use categorical encoding
        channels_last: Whether to use channels-last format
        **kwargs: Additional arguments including file_path and seq_size
    """
    seq_size = kwargs.get('seq_size', 20)
    file_path = kwargs.get('file_path', '../data/2025/nano07_short.csv')
    
    X_train, X_test = load_and_preprocess_data(file_path, seq_size)
    
    input_shape = (seq_size, 4)  # 4 features, seq_size timesteps
    
    return input_shape, X_train, X_test
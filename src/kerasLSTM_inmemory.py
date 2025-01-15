import logging
import numpy as np
from openfl.federated import TensorFlowDataLoader
from .data_utils import load_and_preprocess_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KerasLSTMInMemory(TensorFlowDataLoader):
    """Keras Data Loader for Time-Series Dataset."""

    def __init__(self, data_path, batch_size, timesteps, **kwargs):
        super().__init__(batch_size, **kwargs)

        logger.info(f'Received Data Path: {data_path}')
        
        np.random.seed(42)

        try:
            X_train, X_test = load_and_preprocess_data(data_path, timesteps)
            
            train_size = int(0.8 * len(X_train))
            indices = np.random.permutation(len(X_train))
            train_indices = indices[:train_size]
            valid_indices = indices[train_size:]
            
            self.X_train = X_train[train_indices]
            self.X_valid = X_train[valid_indices]

            self.y_train = self.X_train
            self.y_valid = self.X_valid

        except Exception as e:
            logger.error(f'Error loading data: {e}')
            raise

        self.num_classes = None

        logger.info(f'Train Data Shape: {self.X_train.shape}')
        logger.info(f'Validation Data Shape: {self.X_valid.shape}')
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

def plot_predictions_with_anomalies(data, predictions, columns, dataset_name, seq_size):
   """Plot actual vs predicted values"""
   fig, axes = plt.subplots(len(columns), 1, figsize=(15, 5*len(columns)))
   if len(columns) == 1:
       axes = [axes]
       
   for i, (ax, col) in enumerate(zip(axes, columns)):
       # Plot actual values
       ax.plot(data['datetimestamp'][seq_size:], data[col][seq_size:],
               label='Actual', color='blue', alpha=0.6)
               
       # Plot predicted values
       ax.plot(data['datetimestamp'][seq_size:], predictions[:, i],
               label='Predicted', color='green', alpha=0.6)
               
       ax.set_title(f'{dataset_name}: {col} - Actual vs Predicted')
       ax.set_xlabel('Time')
       ax.set_ylabel('Value')
       ax.legend()
       ax.grid(True, alpha=0.3)
       plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
       
   plt.tight_layout()
   plt.savefig(f'{dataset_name}_predictions.png')
   plt.close()

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

def analyze_dataset(data_path: str, dataset_name: str, model: keras.Model):
   """Analyze dataset and generate predictions visualization"""
   # Load and preprocess data
   data = pd.read_csv(data_path)
   data['datetimestamp'] = pd.to_datetime(data['timestamp'])
   
   columns = [
       'jetson_gpu_usage_percent',
       'jetson_board_temperature_celsius',
       'jetson_cpu_usage_percent',
       'jetson_ram_usage_mb'
    ]
   
   # Initialize scalers with specific parameters
   gpu_scaler = GPUUsageScaler(idle_threshold=1.0, idle_scale=-5.0)
   temp_scaler = StandardScaler()
   cpu_scaler = StandardScaler()
   ram_scaler = StandardScaler()

   # Scale features individually
   scaled_gpu = gpu_scaler.fit_transform(data['jetson_gpu_usage_percent'].values.reshape(-1, 1))
   scaled_temp = temp_scaler.fit_transform(data['jetson_board_temperature_celsius'].values.reshape(-1, 1))
   scaled_cpu = cpu_scaler.fit_transform(data['jetson_cpu_usage_percent'].values.reshape(-1, 1))
   scaled_ram = ram_scaler.fit_transform(data['jetson_ram_usage_mb'].values.reshape(-1, 1))

   # Create scaled DataFrame
   scaled_df = pd.DataFrame({
       'jetson_gpu_usage_percent': scaled_gpu.flatten(),
       'jetson_board_temperature_celsius': scaled_temp.flatten(),
       'jetson_cpu_usage_percent': scaled_cpu.flatten(),
       'jetson_ram_usage_mb': scaled_ram.flatten()
   }, index=data.index)
   
   # Initialize variables
   seq_size = 20
   predictions = np.zeros((len(data) - seq_size, len(columns)))
   
   # Generate predictions using sliding window
   for i in range(len(data) - seq_size):
       sequence = scaled_df.iloc[i:i+seq_size].values.reshape(1, seq_size, len(columns))
       pred = model.predict(sequence, verbose=0)
       predictions[i] = pred[0][0]  # Use the first timestep prediction
   
   # Inverse transform predictions for each feature separately
   predictions_original = np.zeros_like(predictions)
   predictions_original[:, 0] = gpu_scaler.inverse_transform(predictions[:, 0].reshape(-1, 1)).flatten()
   predictions_original[:, 1] = temp_scaler.inverse_transform(predictions[:, 1].reshape(-1, 1)).flatten()
   predictions_original[:, 2] = cpu_scaler.inverse_transform(predictions[:, 2].reshape(-1, 1)).flatten()
   predictions_original[:, 3] = ram_scaler.inverse_transform(predictions[:, 3].reshape(-1, 1)).flatten()
   
   # Generate visualization
   plot_predictions_with_anomalies(data, predictions_original, columns, dataset_name, seq_size)

def main():
   """Main function"""
   # Load model
   model = tf.keras.models.load_model("results/new_custom_metrics/lstm/output_model", compile=False)
   #model = tf.keras.models.load_model("results/new_custom_metrics/bilstm/output_model", compile=False)
   
   # Analyze datasets
   analyze_dataset('data/2025/testdata/Nano07V2_gt.csv', 'V2', model)
   analyze_dataset('data/2025/testdata/Nano07V3_gt.csv', 'V3', model)
   analyze_dataset('data/2025/testdata/Nano08V1_gt.csv', 'V1', model)
   analyze_dataset('data/2025/testdata/Nano07V2_gt_processed.csv', 'V2_processed', model)
   analyze_dataset('data/2025/testdata/Nano07V3_gt_processed.csv', 'V3_processed', model)

if __name__ == "__main__":
   main()
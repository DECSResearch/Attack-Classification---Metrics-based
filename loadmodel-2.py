import importlib.util
import time
from openfl.interface.model import get_model
import joblib
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

def analyze_dataset(data_path, dataset_name):
    """Analyze a single dataset and generate visualizations"""
    # Load and preprocess data
    data = pd.read_csv(data_path)
    data['datetimestamp'] = pd.to_datetime(data['datetimestamp'])
    
    # Create sequences
    seq_size = 20
    combined_X, combined_Y = to_sequence(data[['Hz_mod_anomaly']], data['Hz_mod_anomaly'], seq_size)
    
    # Model prediction and timing
    start_time = time.time()
    combined_predict = model.predict(combined_X)
    end_time = time.time()
    inference_time = end_time - start_time
    
    # Calculate MAPE and threshold
    combined_mape = np.mean(np.abs(combined_predict - combined_X) / combined_X, axis=1) * 100
    max_trainMAPE = np.percentile(combined_mape, 95)
    
    # Create DataFrame with results
    anomaly_df = pd.DataFrame(data[seq_size:])
    anomaly_df['combinedMAPE'] = combined_mape
    anomaly_df['max_trainMAPE'] = max_trainMAPE
    anomaly_df['anomaly'] = anomaly_df['combinedMAPE'] > max_trainMAPE
    
    # Calculate performance metrics
    true_labels = (data['mod_BIN'][seq_size:] != 0).astype(int)
    predicted_labels = (combined_mape > max_trainMAPE).astype(int)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    
    TP = conf_matrix[1, 1]
    FP = conf_matrix[0, 1]
    TN = conf_matrix[0, 0]
    FN = conf_matrix[1, 0]
    
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    # Print results
    print(f"\n=== Results for {dataset_name} ===")
    print(f"Inference time: {inference_time:.4f} seconds")
    print(f"Average time per sample: {(inference_time/len(combined_X))*1000:.4f} ms")
    print(f"Threshold (95th percentile): {max_trainMAPE:.4f}")
    print(f"Accuracy: {accuracy_score(true_labels, predicted_labels)*100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1_score:.4f}")
    
    # Generate key visualizations
    
    # 1. MAPE Timeline with Threshold
    plt.figure(figsize=(12, 6))
    plt.plot(anomaly_df['datetimestamp'], anomaly_df['combinedMAPE'], 
             label='MAPE', alpha=0.7)
    plt.axhline(y=max_trainMAPE, color='r', linestyle='--', 
                label=f'Threshold ({max_trainMAPE:.2f})')
    plt.xlabel('Time')
    plt.ylabel('MAPE (%)')
    plt.title(f'{dataset_name}: Anomaly Detection Timeline')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_anomaly_timeline.png')
    plt.close()
    
    # 2. Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{dataset_name}: Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_confusion_matrix.png')
    plt.close()
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy_score(true_labels, predicted_labels)
    }

def to_sequence(x, y, seq_size=1):
    """Convert data into sequence for LSTM input"""
    x_values = []
    y_values = []
    for i in range(len(x)-seq_size):
        x_values.append(x.iloc[i:(i+seq_size)].values)
        y_values.append(y.iloc[i+seq_size])
    return np.array(x_values), np.array(y_values)

# Load model
model = keras.models.load_model("models-611/bilstm-100-64", compile=False)
#model = keras.models.load_model("models-611/lstm-100-64", compile=False)
#model = keras.models.load_model("models-611/lstm-64-64", compile=False)
#model = keras.models.load_model("models-611/lstm-100-128", compile=False)

# Define datasets
datasets = {
    'V3S1': 'data/Anomalous_data/V3S1.csv',
    'V3S2': 'data/Anomalous_data/V3S2.csv',
    'V3S3': 'data/Anomalous_data/V3S3.csv'
}

# Analyze all datasets and collect results
results = {}
for name, path in datasets.items():
    results[name] = analyze_dataset(path, name)

# Compare results across datasets
metrics = ['accuracy', 'precision', 'recall', 'f1_score']
comparison_data = pd.DataFrame(results).T * 100  # Convert to percentages

# Plot comparison of metrics across datasets
plt.figure(figsize=(12, 6))
comparison_data.plot(kind='bar', width=0.8)
plt.title('Performance Metrics Comparison Across Datasets')
plt.xlabel('Dataset')
plt.ylabel('Percentage (%)')
plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('datasets_comparison.png')
plt.close()
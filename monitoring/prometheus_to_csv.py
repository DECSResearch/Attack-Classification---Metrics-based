import requests
import csv
import os
from datetime import datetime, timedelta

# Prometheus settings
PROMETHEUS_URL = "http://192.168.1.23:9090"  # Your Prometheus URL
START_TIME = int((datetime.now() - timedelta(minutes=10)).timestamp())  # Start time (10 days ago)
END_TIME = int(datetime.now().timestamp())  # End time (now)
STEP = "15"  # 60 seconds step (1-minute intervals)

# Directory to store CSV files
OUTPUT_DIR = "/home/c2sragx03/prometheus/ddata"

# Ensure the directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to query Prometheus for all metrics
def query_all_metrics():
    # Get the list of all active metrics from Prometheus's label values endpoint
    response = requests.get(f"{PROMETHEUS_URL}/api/v1/label/__name__/values")
    metrics = response.json().get("Train_data", [])
    return metrics

# Function to query Prometheus for time-series Train_data over the past 10 days for each metric
def query_metric_data(metric_name):
    params = {
        "query": metric_name,
        "start": START_TIME,
        "end": END_TIME,
        "step": STEP
    }
    response = requests.get(f"{PROMETHEUS_URL}/api/v1/query_range", params=params)
    return response.json()

# Function to save Train_data as CSV
def save_to_csv(data, metric_name):
    # Generate the file path
    file_name = f"prometheus_data_{metric_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    file_path = os.path.join(OUTPUT_DIR, file_name)

    # Extract time-series Train_data
    results = data.get("Train_data", {}).get("result", [])
    
    # Save the Train_data as CSV
    with open(file_path, "w", newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header
        csv_writer.writerow(["timestamp", "value"])

        # Write the rows for each result
        for result in results:
            for value_pair in result.get("values", []):
                timestamp, value = value_pair
                csv_writer.writerow([datetime.utcfromtimestamp(float(timestamp)).strftime('%Y-%m-%d %H:%M:%S'), value])

    print(f"Data for {metric_name} saved to {file_path}")

if __name__ == "__main__":
    # Step 1: Get all metric names
    metrics = query_all_metrics()

    # Step 2: Query each metric's time-series Train_data for the past 10 days and save it to CSV
    for metric in metrics:
        try:
            metric_data = query_metric_data(metric)
            save_to_csv(metric_data, metric)
        except Exception as e:
            print(f"Failed to query or save Train_data for {metric}: {e}")

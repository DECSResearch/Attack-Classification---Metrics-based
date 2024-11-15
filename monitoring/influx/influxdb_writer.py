import requests
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timedelta

# Prometheus settings
PROMETHEUS_URL = "http://prometheus:9090"  # Use Prometheus service name since it's running in docker
START_TIME = int((datetime.now() - timedelta(hours=2)).timestamp())  # Query data for the past 2 hours
END_TIME = int(datetime.now().timestamp())
STEP = "30"  # Query data every 60 seconds

# InfluxDB settings
INFLUXDB_URL= "172.16.233.243:8086"
INFLUXDB_TOKEN= "EkQiFpHFuMOuC2Wdu-9EFVziQ99gZTch5NP7lpYeF3McppKnvTyq7Yn7crPkzgTcsOIYNc-mp23UCYId_JCDyg=="
INFLUXDB_ORG= "testbed-test"
INFLUXDB_BUCKET= "metrics"

# Connect to InfluxDB
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

# Function to query Prometheus for all metric names
def query_all_metrics():
    response = requests.get(f"{PROMETHEUS_URL}/api/v1/label/__name__/values")
    data = response.json()
    return data['data']

# Function to query Prometheus for time-series data of a specific metric
def query_prometheus(metric_name):
    params = {
        "query": metric_name,
        "start": START_TIME,
        "end": END_TIME,
        "step": STEP
    }
    response = requests.get(f"{PROMETHEUS_URL}/api/v1/query_range", params=params)
    data = response.json()
    return data['data']['result']

# Function to write data to InfluxDB
def write_to_influxdb(metric_name, values):
    for value in values:
        timestamp, metric_value = value
        point = Point(metric_name).field("value", float(metric_value)).time(datetime.utcfromtimestamp(float(timestamp)))
        write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
    print(f"Data for {metric_name} written to InfluxDB")

if __name__ == "__main__":
    # Step 1: Get all metric names from Prometheus
    metric_names = query_all_metrics()

    # Step 2: Query each metric and push the data to InfluxDB
    for metric_name in metric_names:
        try:
            result = query_prometheus(metric_name)
            for metric in result:
                values = metric['values']
                write_to_influxdb(metric_name, values)
        except Exception as e:
            print(f"Failed to process metric {metric_name}: {e}")

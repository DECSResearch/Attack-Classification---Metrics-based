import re
import subprocess
import time
from flask import Flask, Response
from prometheus_client import Gauge, generate_latest
import threading
import logging

app = Flask(__name__)

# Define Prometheus metrics
ram_usage_gauge = Gauge('jetson_ram_usage', 'RAM usage in MB', ['type'])
swap_usage_gauge = Gauge('jetson_swap_usage', 'SWAP usage in MB', ['type'])
cpu_usage_gauge = Gauge('jetson_cpu_usage', 'CPU usage per core', ['core'])
emc_freq_gauge = Gauge('jetson_emc_freq', 'EMC frequency usage in %')
gpu_freq_gauge = Gauge('jetson_gpu_freq', 'GPU frequency usage in %')
temperature_gauge = Gauge('jetson_temperature', 'Temperature readings in Celsius', ['sensor'])

def parse_tegrastats_line(line):
    # Regex patterns for different metrics
    ram_regex = re.compile(r'RAM (\d+)/(\d+)MB')
    swap_regex = re.compile(r'SWAP (\d+)/(\d+)MB')
    cpu_regex = re.compile(r'CPU \[([0-9%@, ]+)\]')
    emc_freq_regex = re.compile(r'EMC_FREQ (\d+)%')
    gpu_freq_regex = re.compile(r'GR3D_FREQ (\d+)%')  
    temperature_regex = re.compile(r'(\w+)@([-\d.]+)C')

    # Extract RAM usage
    ram_match = ram_regex.search(line)
    if ram_match:
        used_ram, total_ram = int(ram_match.group(1)), int(ram_match.group(2))
        ram_usage_gauge.labels(type='used').set(used_ram)
        ram_usage_gauge.labels(type='total').set(total_ram)

    # Extract SWAP usage
    swap_match = swap_regex.search(line)
    if swap_match:
        used_swap, total_swap = int(swap_match.group(1)), int(swap_match.group(2))
        swap_usage_gauge.labels(type='used').set(used_swap)
        swap_usage_gauge.labels(type='total').set(total_swap)

    # Extract CPU usage
    cpu_match = cpu_regex.search(line)
    if cpu_match:
        cpu_usages = cpu_match.group(1).split(',')
        for i, usage in enumerate(cpu_usages):
            cpu_percentage = int(usage.split('%')[0].strip())
            cpu_usage_gauge.labels(core=f'cpu{i}').set(cpu_percentage)

    # Extract EMC frequency
    emc_freq_match = emc_freq_regex.search(line)
    if emc_freq_match:
        emc_freq_gauge.set(int(emc_freq_match.group(1)))

    # Extract GPU frequency
    gpu_freq_match = gpu_freq_regex.search(line)
    if gpu_freq_match:
        gpu_freq_gauge.set(int(gpu_freq_match.group(1)))

    # Extract temperature readings
    for sensor, temp in temperature_regex.findall(line):
        try:
            temp_value = float(temp)
            temperature_gauge.labels(sensor=sensor).set(temp_value)
        except ValueError:
            continue

def run_tegrastats():
    """
    Runs tegrastats and parses the output in real time.
    """
    try:
        with subprocess.Popen(['tegrastats', '--interval', '5000'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
            for line in iter(process.stdout.readline, ''):
                parse_tegrastats_line(line)
                time.sleep(1)  # Avoid overwhelming CPU with tight loops
    except Exception as e:
        logging.error(f"Failed to run tegrastats: {e}")

@app.route('/metrics')
def metrics():
    """
    Exposes the metrics in Prometheus format.
    """
    return Response(generate_latest(), mimetype='text/plain')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Start tegrastats parsing in a separate thread
    tegrastats_thread = threading.Thread(target=run_tegrastats, daemon=True)
    tegrastats_thread.start()

    # Start Flask HTTP server for Prometheus to scrape
    app.run(host='0.0.0.0', port=8085)

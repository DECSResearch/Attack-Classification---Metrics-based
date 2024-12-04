import tensorflow as tf
import psutil
import time
import argparse
import numpy as np
import random


# Mode 1: Interval (10min stress, 10min rest)
# python3 script.py --device gpu --usage 40 --mode 1 --duration 7200 --stress-time 20 --rest-time 20

# Mode 2: Continuous
# python3 script.py --device gpu --usage 40 --mode 2 --duration 3600

# Mode 3: Random intervals
# python3 script.py --device gpu --usage 40 --mode 3 --duration 3600

def stress_gpu(target_usage, duration):
    """GPU stress test to maintain specified usage"""
    if not tf.config.list_physical_devices('GPU'):
        print("No GPU detected")
        return
        
    start_time = time.time()
    while time.time() - start_time < duration:
        with tf.device('/GPU:0'):
            a = tf.random.normal([5000, 5000])
            b = tf.random.normal([5000, 5000])
            c = tf.matmul(a, b)
            
        if float(tf.config.experimental.get_memory_usage('GPU:0')) > target_usage:
            time.sleep(0.1)

def stress_cpu(target_usage, duration):
    """CPU stress test to maintain specified usage"""
    start_time = time.time()
    while time.time() - start_time < duration:
        if psutil.cpu_percent() > target_usage:
            time.sleep(0.1)
        else:
            with tf.device('/CPU:0'):
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])
                c = tf.matmul(a, b)

def interval_stress(device, target_usage, stress_duration, rest_duration, cycles):
    """Mode 1: Interval stress testing"""
    print(f"Starting interval stress test: {cycles} cycles")
    for i in range(cycles):
        print(f"Cycle {i+1}: Stress phase")
        if device == 'gpu':
            stress_gpu(target_usage, stress_duration)
        else:
            stress_cpu(target_usage, stress_duration)
            
        print(f"Cycle {i+1}: Rest phase")
        time.sleep(rest_duration)

def random_stress(device, target_usage, total_duration, min_interval=60, max_interval=300):
    """Mode 3: Random interval stress testing"""
    start_time = time.time()
    while time.time() - start_time < total_duration:
        if random.choice([True, False]):
            duration = random.randint(min_interval, max_interval)
            print(f"Starting random stress for {duration} seconds")
            if device == 'gpu':
                stress_gpu(target_usage, duration)
            else:
                stress_cpu(target_usage, duration)
        else:
            rest_time = random.randint(min_interval, max_interval)
            print(f"Resting for {rest_time} seconds")
            time.sleep(rest_time)

def main():
    parser = argparse.ArgumentParser(description='GPU/CPU Stress Testing Tool')
    parser.add_argument('--device', choices=['gpu', 'cpu'], required=True, help='Select device')
    parser.add_argument('--usage', type=float, required=True, help='Target usage (0-100)')
    parser.add_argument('--mode', type=int, choices=[1,2,3], required=True, help='1:Interval, 2:Continuous, 3:Random')
    parser.add_argument('--duration', type=int, required=True, help='Total duration in seconds')
    parser.add_argument('--stress-time', type=int, help='Stress duration for interval mode (seconds)')
    parser.add_argument('--rest-time', type=int, help='Rest duration for interval mode (seconds)')
    
    args = parser.parse_args()
    
    if args.mode == 1:
        if not args.stress_time or not args.rest_time:
            print("Stress time and rest time required for interval mode")
            return
        cycles = args.duration // (args.stress_time + args.rest_time)
        interval_stress(args.device, args.usage, args.stress_time, args.rest_time, cycles)
    elif args.mode == 2:
        if args.device == 'gpu':
            stress_gpu(args.usage, args.duration)
        else:
            stress_cpu(args.usage, args.duration)
    else:
        random_stress(args.device, args.usage, args.duration)

if __name__ == "__main__":
    main()

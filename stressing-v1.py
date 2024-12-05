import tensorflow as tf
import psutil
import time
import argparse
import numpy as np
import random
from multiprocessing import Process, Value, Lock, active_children, Pipe
import os
import signal
import sys
import threading

# for some unknown reasons, if you use "X" as usage in the cpu stress, you get "X + 20" as the stressing usage
# The usage function of cpu and gpu both don't function well on the testbed

# sudo docker run -it --rm --network host --name stressing --rm --runtime=nvidia --gpus all -e NVIDIA_VISIBLE_DEVICES=all changcunlei/attack-simulation:latest bash

# Mode 1: Interval (10min stress, 10min rest)
# python3 stressing.py --device gpu --usage 40 --mode 1 --duration 7200 --stress-time 20 --rest-time 20

# Mode 2: Continuous
# python3 stressing.py --device gpu --usage 40 --mode 2 --duration 3600

# Mode 3: Random intervals
# python3 stressing.py --device gpu --usage 40 --mode 3 --duration 3600

def cpu_worker(target_usage, shared_usage, lock, usage_pipe):
    """CPU worker with fine-grained control"""
    current_target = target_usage
    while True:
        if usage_pipe.poll():
            current_target = usage_pipe.recv()
            
        start_time = time.time()
        while time.time() - start_time < 0.1:
            pass
            
        with lock:
            current_usage = shared_usage.value
            if current_usage > current_target:
                time.sleep(0.1)
            else:
                time.sleep(0.01)

def stress_cpu(target_usage, duration, usage_pipe=None):
    """CPU stress test with shared memory for usage control"""
    shared_usage = Value('d', 0.0)
    lock = Lock()
    local_pipe = usage_pipe if usage_pipe else Pipe()
    parent_conn = local_pipe[0] if isinstance(local_pipe, tuple) else usage_pipe
    child_conn = local_pipe[1] if isinstance(local_pipe, tuple) else usage_pipe
    
    def monitor_usage():
        while True:
            usage = psutil.cpu_percent(interval=0.5)
            with lock:
                shared_usage.value = usage
    
    monitor_proc = Process(target=monitor_usage)
    monitor_proc.start()
    
    processes = []
    core_count = psutil.cpu_count()
    worker_count = max(1, int(core_count * target_usage / 100))
    
    for _ in range(worker_count):
        p = Process(target=cpu_worker, args=(target_usage, shared_usage, lock, child_conn))
        p.start()
        processes.append(p)
    
    try:
        time.sleep(duration)
    finally:
        monitor_proc.terminate()
        for p in processes:
            p.terminate()
        
    return parent_conn if not usage_pipe else None

def gpu_worker(batch_size, complexity, usage_pipe):
    """GPU worker optimized for Jetson Nano"""
    current_target = None
    while True:
        if usage_pipe.poll():
            current_target = usage_pipe.recv()
            
        with tf.device('/GPU:0'):
            # Removed mixed precision policy
            for _ in range(complexity):
                a = tf.random.normal([batch_size, batch_size])
                b = tf.random.normal([batch_size, batch_size])
                c = tf.matmul(a, b)
                _ = c.numpy()

def stress_gpu(target_usage, duration, usage_pipe=None):
    """GPU stress test with dynamic load adjustment"""
    if not tf.config.list_physical_devices('GPU'):
        print("No GPU detected")
        return
        
    local_pipe = usage_pipe if usage_pipe else Pipe()
    parent_conn = local_pipe[0] if isinstance(local_pipe, tuple) else usage_pipe
    child_conn = local_pipe[1] if isinstance(local_pipe, tuple) else usage_pipe
    
    batch_size = 1000
    complexity = 2
    current_target = target_usage
    
    start_time = time.time()
    while time.time() - start_time < duration:
        try:
            if parent_conn.poll():
                current_target = parent_conn.recv()
                print(f"GPU target usage updated to: {current_target}%")
                
            gpu_util = float(os.popen('nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits').read())
            
            if gpu_util < current_target - 5:
                batch_size = min(batch_size + 100, 5000)
                complexity = min(complexity + 1, 5)
            elif gpu_util > current_target + 5:
                batch_size = max(batch_size - 100, 500)
                complexity = max(complexity - 1, 1)
                
            gpu_worker(batch_size, complexity, child_conn)
            
        except Exception as e:
            print(f"Error in GPU stress test: {str(e)}")
            time.sleep(1)
            
    return parent_conn if not usage_pipe else None

def interval_stress(device, target_usage, stress_duration, rest_duration, cycles):
    """Mode 1: Interval stress testing"""
    print(f"Starting interval stress test: {cycles} cycles")
    parent_conn, child_conn = Pipe()
    
    def usage_input():
        while True:
            try:
                new_usage = float(input("Enter new usage (0-100): "))
                if 0 <= new_usage <= 100:
                    parent_conn.send(new_usage)
                    print(f"Usage updated to {new_usage}%")
                else:
                    print("Usage must be between 0 and 100")
            except ValueError:
                print("Please enter a valid number")
            except Exception as e:
                if not isinstance(e, EOFError):
                    print(f"Error updating usage: {e}")
                break
    
    input_thread = threading.Thread(target=usage_input, daemon=True)
    input_thread.start()
    
    for i in range(cycles):
        print(f"Cycle {i+1}: Stress phase")
        if device == 'gpu':
            stress_gpu(target_usage, stress_duration, child_conn)
        else:
            stress_cpu(target_usage, stress_duration, child_conn)
            
        print(f"Cycle {i+1}: Rest phase")
        time.sleep(rest_duration)

def continuous_stress(device, target_usage, duration):
    """Mode 2: Continuous stress testing"""
    print(f"Starting continuous stress test for {duration} seconds")
    parent_conn, child_conn = Pipe()
    
    def usage_input():
        while True:
            try:
                new_usage = float(input("Enter new usage (0-100): "))
                if 0 <= new_usage <= 100:
                    parent_conn.send(new_usage)
                    print(f"Usage updated to {new_usage}%")
                else:
                    print("Usage must be between 0 and 100")
            except ValueError:
                print("Please enter a valid number")
            except Exception as e:
                if not isinstance(e, EOFError):
                    print(f"Error updating usage: {e}")
                break
    
    input_thread = threading.Thread(target=usage_input, daemon=True)
    input_thread.start()
    
    if device == 'gpu':
        stress_gpu(target_usage, duration, child_conn)
    else:
        stress_cpu(target_usage, duration, child_conn)

def random_stress(device, target_usage, total_duration, min_interval=5, max_interval=15):
    """Mode 3: Random interval stress testing"""
    parent_conn, child_conn = Pipe()
    
    def usage_input():
        while True:
            try:
                new_usage = float(input("Enter new usage (0-100): "))
                if 0 <= new_usage <= 100:
                    parent_conn.send(new_usage)
                    print(f"Usage updated to {new_usage}%")
                else:
                    print("Usage must be between 0 and 100")
            except ValueError:
                print("Please enter a valid number")
            except Exception as e:
                if not isinstance(e, EOFError):
                    print(f"Error updating usage: {e}")
                break
    
    input_thread = threading.Thread(target=usage_input, daemon=True)
    input_thread.start()
    
    start_time = time.time()
    while time.time() - start_time < total_duration:
        if random.choice([True, False]):
            duration = random.randint(min_interval, max_interval)
            print(f"Starting random stress for {duration} seconds")
            if device == 'gpu':
                stress_gpu(target_usage, duration, child_conn)
            else:
                stress_cpu(target_usage, duration, child_conn)
        else:
            rest_time = random.randint(min_interval, max_interval)
            print(f"Resting for {rest_time} seconds")
            time.sleep(rest_time)

def sigint_handler(signum, frame):
    """Handle keyboard interrupt"""
    procs = active_children()
    for p in procs:
        p.terminate()
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, sigint_handler)
    
    parser = argparse.ArgumentParser(description='GPU/CPU Stress Testing Tool')
    parser.add_argument('--device', choices=['gpu', 'cpu'], required=True, help='Select device')
    parser.add_argument('--usage', type=float, required=True, help='Target usage (0-100)')
    parser.add_argument('--mode', type=int, choices=[1,2,3], required=True, help='1:Interval, 2:Continuous, 3:Random')
    parser.add_argument('--duration', type=int, required=True, help='Total duration in seconds')
    parser.add_argument('--stress-time', type=int, help='Stress duration for interval mode (seconds)')
    parser.add_argument('--rest-time', type=int, help='Rest duration for interval mode (seconds)')
    
    args = parser.parse_args()
    
    if args.usage < 0 or args.usage > 100:
        print("Usage must be between 0 and 100")
        return
        
    try:
        if args.mode == 1:
            if not args.stress_time or not args.rest_time:
                print("Stress time and rest time required for interval mode")
                return
            cycles = args.duration // (args.stress_time + args.rest_time)
            interval_stress(args.device, args.usage, args.stress_time, args.rest_time, cycles)
        elif args.mode == 2:
            continuous_stress(args.device, args.usage, args.duration)
        else:
            random_stress(args.device, args.usage, args.duration)
    except KeyboardInterrupt:
        print("\nStopping stress test...")
        sys.exit(0)

if __name__ == "__main__":
    main()

import sys

# for some unknown reasons, if you use "X" as usage in the cpu stress, you get "X + 20" as the stressing usage
# The usage function of cpu and gpu both don't function well on the testbed

# sudo docker run -it --rm --network host --name stressing --rm --runtime=nvidia --gpus all -e NVIDIA_VISIBLE_DEVICES=all changcunlei/attack-simulation:latest bash

# Mode 1: Interval (10min stress, 10min rest)
# python3 stressing.py --device gpu --usage 40 --mode 1 --duration 7200 --stress-time 20 --rest-time 20
# Mode 2: Continuous
# python3 stressing.py --device gpu --usage 40 --mode 2 --duration 3600
# Mode 3: Random intervals
# python3 stressing.py --device gpu --usage 40 --mode 3 --duration 3600
def loop_process(conn, affinity, check_usage):
    """Single CPU core stress function"""
    proc = psutil.Process()
    proc.cpu_affinity([affinity])
    msg = f"Process ID: {proc.pid} CPU: {affinity}"
    conn.send(msg)
    conn.close()
    
    while True:
        if check_usage and psutil.cpu_percent() > 100:
            time.sleep(0.05)
        1*1
def stress_cpu(target_usage, duration):
    """Enhanced CPU stress test using process affinity"""
    total_cores = psutil.cpu_count(logical=True)
    cores_to_use = int((target_usage * total_cores) / 100)
    fractional_part = (target_usage * total_cores / 100) - cores_to_use
    
    processes = []
    connections = []
    
    # Start processes for full core usage
    for i in range(max(0, cores_to_use - 1)):
        parent_conn, child_conn = Pipe()
        p = Process(target=loop_process, args=(child_conn, i, False))
        p.start()
        processes.append(p)
        connections.append(parent_conn)
    
    # Last full core with usage checking
    if cores_to_use > 0:
        parent_conn, child_conn = Pipe()
        p = Process(target=loop_process, args=(child_conn, cores_to_use - 1, True))
        p.start()
        processes.append(p)
        connections.append(parent_conn)
    
    # Handle fractional core usage if needed
    if fractional_part > 0:
        parent_conn, child_conn = Pipe()
        p = Process(target=loop_process, args=(child_conn, cores_to_use, True))
        p.start()
        processes.append(p)
        connections.append(parent_conn)
    
    # Print process information
    for conn in connections:
        try:
            print(conn.recv())
        except EOFError:
            continue
    
    # Wait for duration
    time.sleep(duration)
    
    # Cleanup
    for p in processes:
        p.terminate()
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
def random_stress(device, target_usage, total_duration, min_interval=5, max_interval=15):
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
def sigint_handler(signum, frame):
    """Handle keyboard interrupt"""
    procs = active_children()
    for p in procs:
        p.terminate()
    os._exit(1)
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

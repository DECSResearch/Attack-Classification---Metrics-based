# Jetson Device Stress Testing Tool

A comprehensive stress testing tool for NVIDIA Jetson devices that allows precise control over CPU and GPU load levels with multiple testing modes.

## Important Implementation Note
- you may need to run "sed -i 's/\r$//' stressing.sh" before running the script

**CPU Stress Testing:**
- Can be run directly on the host machine
- No need for Docker - run the script directly on Jetson Nano
- Provides better performance and more accurate load control

**GPU Stress Testing:**
- If you run the GPU stress testing outside Docker, make sure the CUDA environment is configured properly
- Docker container provides pre-configured environment with:
  - TensorFlow
  - CUDA
  - Required GPU drivers
  - All necessary dependencies

## Docker Setup for GPU Testing

```bash
# Run the Docker container
sudo docker run -it --rm \
    --network host \
    --name stressing \
    --runtime=nvidia \
    -v /usr/bin/tegrastats:/usr/bin/tegrastats \
    --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    changcunlei/attack-simulation:latest \
    bash

# Inside container, run the stress test script
./stressing.sh
```

## Features

- CPU stress testing with configurable load levels (0-100%)
- GPU stress testing with predefined load levels (10-100% in steps of 10)
- Three testing modes: continuous, interval, and random
- Real-time CPU usage monitoring
- Configurable duration and intervals
- Clean process management with proper cleanup

## Usage

### Basic Commands

```bash
cpu <level>      # Set CPU stress level (0-100%)
gpu <level>      # Set GPU stress level (10,20,...,100)
mode <type>      # Set mode (continuous/interval/random)
duration <N>     # Set total test duration in seconds
stress-time <N>  # Set stress phase duration for interval mode
rest-time <N>    # Set rest phase duration for interval mode
start           # Start the stress test
status          # Display current configuration
help            # Show available commands
exit            # Exit the program
```

### Testing Modes

1. **Continuous Mode**
   - Maintains constant stress level for the specified duration
   - Example: `mode continuous; duration 300; cpu 70; start`

2. **Interval Mode**
   - Alternates between stress and rest periods
   - Configurable stress and rest durations
   - Example: `mode interval; duration 600; stress-time 30; rest-time 30; gpu 80; start`

3. **Random Mode**
   - Randomly alternates between stress and rest
   - Random intervals between 5-30 seconds (you can configure that in the code)
   - Example: `mode random; duration 900; cpu 90; start`

## Example Workflows

### CPU Stress Testing (Run on Host)
```bash
# Run directly on Jetson Nano
./stressing.sh

# Then in the script:
cpu 50
mode continuous
duration 300
start
```

### GPU Stress Testing (Run in Docker)
```bash
# 1. First start Docker container
sudo docker run -it --rm --runtime=nvidia --gpus all changcunlei/attack-simulation:latest bash

# 2. Then inside container run the script:
./stressing.sh

# 3. Configure GPU test:
gpu 80
mode interval
duration 600
stress-time 60
rest-time 30
start
```

## Technical Details

### CPU Stress Implementation
- Uses work/sleep cycles to achieve precise CPU load levels
- Implements microsecond-level timing control
- Scales across all available CPU cores
- Formula: `work_time = interval * target_load / 100`

### GPU Stress Implementation
- Uses TensorFlow for matrix multiplication operations
- Predefined matrix sizes for different load levels, you can change them:
```python
GPU_PARAMS = {
    "10": "50x50 matrix",
    "20": "184x184 matrix",
    "30": "265x265 matrix",
    ...
    "100": "650x650 matrix"
}
```

### Process Management
- Tracks all CPU stress processes in `CPU_PIDS` array
- Manages GPU process with `GPU_PID`
- Implements proper cleanup on exit/interrupt
- Uses trap for SIGINT and SIGTERM signals

## Safety Features

- Proper process cleanup on exit
- Signal handling for clean termination
- Resource usage monitoring
- Input validation for all parameters
- Automatic process termination between mode changes

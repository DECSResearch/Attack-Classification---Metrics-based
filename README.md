# Jetson Device Stress Testing Tool

A comprehensive stress testing tool for NVIDIA Jetson devices that allows precise control over CPU and GPU load levels with multiple testing modes.

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
   - Random intervals between 5-30 seconds (can be configured in the code)
   - Example: `mode random; duration 900; cpu 90; start`

## Technical Details

### CPU Stress Implementation
- Uses work/sleep cycles to achieve precise CPU load levels
- Implements microsecond-level timing control
- Scales across all available CPU cores
- Formula: `work_time = interval * target_load / 100`

### GPU Stress Implementation
- Uses TensorFlow for matrix multiplication operations
- Predefined matrix sizes for different load levels:
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

### Monitoring
- Real-time CPU usage monitoring via `top`
- Status command shows:
  - Current CPU/GPU levels
  - Active mode
  - Duration settings
  - Current CPU usage

## Error Handling

The script includes error checking for:
- Invalid CPU/GPU levels
- Missing duration settings
- Incomplete interval mode settings
- Process cleanup on exit

## Performance Notes

1. **CPU Load Control**
   - Achieves precise load levels through microsecond timing
   - Minimal system impact when idle
   - Scales efficiently across all cores

2. **GPU Load Control**
   - Uses calibrated matrix sizes for different load levels
   - Includes sleep intervals for fine-tuning
   - Efficient TensorFlow implementation

## Example Workflows

1. Basic CPU Test:
```bash
cpu 50
mode continuous
duration 300
start
```

2. Interval GPU Test:
```bash
gpu 80
mode interval
duration 600
stress-time 60
rest-time 30
start
```

3. Random CPU Test:
```bash
cpu 70
mode random
duration 900
start
```

## Safety Features

- Proper process cleanup on exit
- Signal handling for clean termination
- Resource usage monitoring
- Input validation for all parameters
- Automatic process termination between mode changes

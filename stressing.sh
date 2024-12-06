#!/bin/bash

# Global variables
CURRENT_CPU_LEVEL=0
CURRENT_GPU_LEVEL=0
CURRENT_MODE="continuous"
CPU_PIDS=()
GPU_PID=""

# CPU stress levels (adjust these values based on testing)
declare -A CPU_PARAMS=(
    [10]="100;0.9"    # iterations;sleep_time
    [20]="1000000;0.8"
    [30]="1000000;0.7"
    [40]="1000000;0.6"
    [50]="1000000;0.5"
    [60]="1000000;0.4"
    [70]="1000000;0.3"
    [80]="1000000;0.2"
    [90]="1000000;0.1"
    [100]="1000000;0"
)

# GPU stress levels (tested on nano 13)
declare -A GPU_PARAMS=(
    [10]="50;0"     # matrix_size;sleep_time
    [20]="184;0"
    [30]="265;0.0001"
    [40]="294;0"
    [50]="356;0"
    [60]="388;0.0001"
    [70]="576;0"
    [80]="618;0"
    [90]="650;0.0001"
    [100]="650;0"
)

# Cleanup function
cleanup() {
    echo "Cleaning up processes..."
    for pid in "${CPU_PIDS[@]}"; do
        kill -9 $pid 2>/dev/null
    done
    [ ! -z "$GPU_PID" ] && kill -9 $GPU_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# CPU stress function
stress_cpu() {
    local level=$1
    local params=${CPU_PARAMS[$level]}
    local iterations=${params%;*}
    local sleep_time=${params#*;}
    
    for pid in "${CPU_PIDS[@]}"; do
        kill -9 $pid 2>/dev/null
    done
    CPU_PIDS=()
    
    local num_cores=$(nproc)
    for ((i=0; i<num_cores; i++)); do
        (
            while true; do
                for ((j=0; j<iterations; j++)); do
                    : # No-op
                done
                sleep $sleep_time
            done
        ) &
        CPU_PIDS+=($!)
    done
}

# GPU stress function
stress_gpu() {
    local level=$1
    local params=${GPU_PARAMS[$level]}
    local matrix_size=${params%;*}
    local sleep_time=${params#*;}
    
    [ ! -z "$GPU_PID" ] && kill -9 $GPU_PID 2>/dev/null
    
    (python3 - << EOF
import tensorflow as tf
import time

while True:
    with tf.device('/GPU:0'):
        a = tf.random.normal([$matrix_size, $matrix_size])
        b = tf.random.normal([$matrix_size, $matrix_size])
        c = tf.matmul(a, b)
    time.sleep($sleep_time)
EOF
    ) &
    GPU_PID=$!
}

# Main loop
handle_user_input() {
    while true; do
        read -p "Enter command (cpu|gpu [10|20|30|40|50|60|70|80|90|100] or mode [continuous|interval|random]): " cmd arg1
        
        case $cmd in
            cpu)
                if [[ ${CPU_PARAMS[$arg1]+_} ]]; then
                    CURRENT_CPU_LEVEL=$arg1
                    stress_cpu $arg1
                    echo "CPU level set to: $arg1%"
                else
                    echo "Invalid CPU level. Use: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100"
                fi
                ;;
            gpu)
                if [[ ${GPU_PARAMS[$arg1]+_} ]]; then
                    CURRENT_GPU_LEVEL=$arg1
                    stress_gpu $arg1
                    echo "GPU level set to: $arg1%"
                else
                    echo "Invalid GPU level. Use: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100"
                fi
                ;;
            status)
                echo "Current levels:"
                echo "CPU: $CURRENT_CPU_LEVEL%"
                echo "GPU: $CURRENT_GPU_LEVEL%"
                ;;
            exit)
                cleanup
                ;;
            *)
                echo "Commands:"
                echo "  cpu <level>   - Set CPU level (10-100, step 10)"
                echo "  gpu <level>   - Set GPU level (10-100, step 10)"
                echo "  status        - Show current levels"
                echo "  exit          - Exit program"
                ;;
        esac
    done
}

echo "Jetson Device Stress Test Tool"
echo "Use Ctrl+C to exit"

handle_user_input

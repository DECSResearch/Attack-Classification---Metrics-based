#!/bin/bash

# Global variables
CURRENT_CPU_LEVEL=0
CURRENT_GPU_LEVEL=0
CURRENT_MODE="continuous"
CPU_PIDS=()
GPU_PID=""
DURATION=60
STRESS_TIME=10
REST_TIME=10

# Matrix size parameters
declare -A CPU_PARAMS=(
    [10]="50;0.1"     # matrix_size;sleep_time
    [20]="184;0.1"
    [30]="265;0.1"
    [40]="294;0.1"
    [50]="356;0.1"
    [60]="388;0.1"
    [70]="576;0.1"
    [80]="618;0.1"
    [90]="650;0.1"
    [100]="650;0"
)

declare -A GPU_PARAMS=(
    [10]="50;0"
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
    local matrix_size=${params%;*}
    local sleep_time=${params#*;}
    
    for pid in "${CPU_PIDS[@]}"; do
        kill -9 $pid 2>/dev/null
    done
    CPU_PIDS=()
    
    local num_cores=$(nproc)
    for ((i=0; i<num_cores; i++)); do
        (python3 - << EOF
import numpy as np
import os
import time

# Set environment variable to control threading
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

while True:
    a = np.random.normal(size=($matrix_size, $matrix_size))
    b = np.random.normal(size=($matrix_size, $matrix_size))
    c = np.matmul(a, b)
    time.sleep($sleep_time)
EOF
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

# Mode control functions
stop_stress() {
    local device=$1
    
    if [ "$device" == "cpu" ]; then
        for pid in "${CPU_PIDS[@]}"; do
            kill -9 $pid 2>/dev/null
        done
        CPU_PIDS=()
    else
        [ ! -z "$GPU_PID" ] && kill -9 $GPU_PID 2>/dev/null
        GPU_PID=""
    fi
}

run_interval_mode() {
    local device=$1
    local level=$2
    local total_duration=$3
    local stress_time=$4
    local rest_time=$5
    
    local cycles=$((total_duration / (stress_time + rest_time)))
    echo "Starting interval mode: $cycles cycles"
    local start_time=$(date +%s)
    
    while [ $(($(date +%s) - start_time)) -lt $total_duration ]; do
        echo "Starting stress phase"
        if [ "$device" == "cpu" ]; then
            stress_cpu $level
        else
            stress_gpu $level
        fi
        
        sleep $stress_time
        stop_stress $device
        
        echo "Starting rest phase"
        sleep $rest_time
    done
}

run_random_mode() {
    local device=$1
    local level=$2
    local total_duration=$3
    local min_interval=60
    local max_interval=300
    
    echo "Starting random mode for $total_duration seconds"
    local start_time=$(date +%s)
    
    while [ $(($(date +%s) - start_time)) -lt $total_duration ]; do
        if [ $((RANDOM % 2)) -eq 0 ]; then
            local duration=$((RANDOM % (max_interval - min_interval + 1) + min_interval))
            echo "Starting random stress for $duration seconds"
            if [ "$device" == "cpu" ]; then
                stress_cpu $level
            else
                stress_gpu $level
            fi
            sleep $duration
            stop_stress $device
        else
            local rest_time=$((RANDOM % (max_interval - min_interval + 1) + min_interval))
            echo "Resting for $rest_time seconds"
            sleep $rest_time
        fi
    done
}

run_continuous_mode() {
    local device=$1
    local level=$2
    local duration=$3
    
    echo "Starting continuous mode for $duration seconds"
    if [ "$device" == "cpu" ]; then
        stress_cpu $level
    else
        stress_gpu $level
    fi
    
    sleep $duration
    stop_stress $device
}

# Main command handler
handle_user_input() {
    while true; do
        read -p "Enter command (cpu|gpu [10-100] | mode [continuous|interval|random] | duration N | stress-time N | rest-time N | start | status | help): " cmd arg1
        
        case $cmd in
            cpu)
                if [[ ${CPU_PARAMS[$arg1]+_} ]]; then
                    CURRENT_CPU_LEVEL=$arg1
                    echo "CPU level set to: $arg1%"
                else
                    echo "Invalid CPU level. Use: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100"
                fi
                ;;
            gpu)
                if [[ ${GPU_PARAMS[$arg1]+_} ]]; then
                    CURRENT_GPU_LEVEL=$arg1
                    echo "GPU level set to: $arg1%"
                else
                    echo "Invalid GPU level. Use: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100"
                fi
                ;;
            mode)
                case $arg1 in
                    continuous|interval|random)
                        CURRENT_MODE=$arg1
                        echo "Mode set to: $arg1"
                        ;;
                    *)
                        echo "Invalid mode. Use: continuous, interval, or random"
                        ;;
                esac
                ;;
            duration)
                DURATION=$arg1
                echo "Duration set to: $arg1 seconds"
                ;;
            stress-time)
                STRESS_TIME=$arg1
                echo "Stress time set to: $arg1 seconds"
                ;;
            rest-time)
                REST_TIME=$arg1
                echo "Rest time set to: $arg1 seconds"
                ;;
            start)
                if [ $DURATION -eq 0 ]; then
                    echo "Please set duration first"
                    continue
                fi
                
                case $CURRENT_MODE in
                    continuous)
                        if [ $CURRENT_CPU_LEVEL -gt 0 ]; then
                            run_continuous_mode "cpu" $CURRENT_CPU_LEVEL $DURATION
                        elif [ $CURRENT_GPU_LEVEL -gt 0 ]; then
                            run_continuous_mode "gpu" $CURRENT_GPU_LEVEL $DURATION
                        else
                            echo "Please set CPU or GPU level first"
                        fi
                        ;;
                    interval)
                        if [ $STRESS_TIME -eq 0 ] || [ $REST_TIME -eq 0 ]; then
                            echo "Please set stress-time and rest-time first"
                            continue
                        fi
                        if [ $CURRENT_CPU_LEVEL -gt 0 ]; then
                            run_interval_mode "cpu" $CURRENT_CPU_LEVEL $DURATION $STRESS_TIME $REST_TIME
                        elif [ $CURRENT_GPU_LEVEL -gt 0 ]; then
                            run_interval_mode "gpu" $CURRENT_GPU_LEVEL $DURATION $STRESS_TIME $REST_TIME
                        else
                            echo "Please set CPU or GPU level first"
                        fi
                        ;;
                    random)
                        if [ $CURRENT_CPU_LEVEL -gt 0 ]; then
                            run_random_mode "cpu" $CURRENT_CPU_LEVEL $DURATION
                        elif [ $CURRENT_GPU_LEVEL -gt 0 ]; then
                            run_random_mode "gpu" $CURRENT_GPU_LEVEL $DURATION
                        else
                            echo "Please set CPU or GPU level first"
                        fi
                        ;;
                esac
                ;;
            status)
                echo "Current configuration:"
                echo "CPU level: $CURRENT_CPU_LEVEL%"
                echo "GPU level: $CURRENT_GPU_LEVEL%"
                echo "Mode: $CURRENT_MODE"
                echo "Duration: $DURATION seconds"
                echo "Stress time: $STRESS_TIME seconds"
                echo "Rest time: $REST_TIME seconds"
                ;;
            help)
                echo "Commands:"
                echo "  cpu <level>      - Set CPU level (10-100, step 10)"
                echo "  gpu <level>      - Set GPU level (10-100, step 10)"
                echo "  mode <type>      - Set mode (continuous, interval, random)"
                echo "  duration <N>     - Set total duration in seconds"
                echo "  stress-time <N>  - Set stress duration for interval mode"
                echo "  rest-time <N>    - Set rest duration for interval mode"
                echo "  start           - Start stress test with current settings"
                echo "  status          - Show current settings"
                echo "  help            - Show this help"
                echo "  exit            - Exit program"
                ;;
            exit)
                cleanup
                ;;
            *)
                echo "Unknown command. Type 'help' for available commands."
                ;;
        esac
    done
}

echo "Jetson Device Stress Test Tool"
echo "Type 'help' for available commands"
echo "Use Ctrl+C to exit"
handle_user_input

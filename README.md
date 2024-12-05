The CPU stress test is based on fully utilizing different cores of the CPU. For example, if the CPU has a total of 6 cores, 50% usage could indicate that 3 cores are being fully utilized.

The GPU part is unfinished.

Use sudo docker run -it --rm --network host --name stressing --rm --runtime=nvidia -v /usr/bin/tegrastats:/usr/bin/tegrastats --gpus all -e NVIDIA_VISIBLE_DEVICES=all changcunlei/attack-simulation:latest bash
So you have access to tegrastats.

sudo docker run -it --rm --network host --name stressing --rm --runtime=nvidia --gpus all -e NVIDIA_VISIBLE_DEVICES=all changcunlei/attack-simulation:latest bash

Mode 1: Interval (10s stress, 10s rest)
python3 stressing.py --device gpu --usage 40 --mode 1 --duration 60 --stress-time 10 --rest-time 10

Mode 2: Continuous
python3 stressing.py --device gpu --usage 40 --mode 2 --duration 60

Mode 3: Random intervals
python3 stressing.py --device gpu --usage 40 --mode 3 --duration 60

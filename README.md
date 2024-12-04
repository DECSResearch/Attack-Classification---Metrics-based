

The usage of the cpu and gpu don't work well on the testbed



sudo docker run -it --rm --network host --name stressing --rm --runtime=nvidia --gpus all -e NVIDIA_VISIBLE_DEVICES=all changcunlei/attack-simulation:latest bash

Mode 1: Interval (10s stress, 10s rest)
python3 stressing.py --device gpu --usage 40 --mode 1 --duration 60 --stress-time 10 --rest-time 10

Mode 2: Continuous
python3 stressing.py --device gpu --usage 40 --mode 2 --duration 60

Mode 3: Random intervals
python3 stressing.py --device gpu --usage 40 --mode 3 --duration 60

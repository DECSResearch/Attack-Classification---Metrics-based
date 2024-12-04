sudo docker run -it --rm --network host --name openflclient1 --rm --runtime=nvidia --gpus all -e NVIDIA_VISIBLE_DEVICES=all changcunlei/attack-simulation:latest bash

Mode 1: Interval (10min stress, 10min rest)
python3 stressing.py --device gpu --usage 40 --mode 1 --duration 7200 --stress-time 600 --rest-time 600

Mode 2: Continuous
python3 stressing.py --device gpu --usage 40 --mode 2 --duration 3600

Mode 3: Random intervals
python3 stressing.py --device gpu --usage 40 --mode 3 --duration 3600

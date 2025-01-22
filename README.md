# Attack-Classification---Metrics-based
This project focuses on classifying attacks based on the hardware metrices of IoT devices


docker build --platform=linux/arm64 -t hamzakarim07/flwr_server_metrics:latest -f FL_server/docker/Dockerfile .

python3 client.py --ip=192.168.1.24 --folder=Nano08 --id=2

python3 server.py
# Attack-Classification---Metrics-based
This project focuses on classifying attacks based on the hardware metrices of IoT devices

## Build Docker Images
docker build --platform=linux/arm64 -t hamzakarim07/flwr_server_metrics:latest -f FL_server/docker/Dockerfile .

## Run Client Script
python3 client.py --ip=192.168.1.24 --folder=Nano08 --id=2

## Run Server Script
python3 server.py

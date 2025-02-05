# OpenFL

A Framework for Federated Learning

## Prerequisites

Before running the framework, execute the following command to remove Docker RAM usage limitations:
```bash
ulimit -s unlimited
```

For more detailed information, please refer to the [official OpenFL documentation](https://openfl.readthedocs.io/en/latest/).

## Quick Start

### Launch the Container

Run the following command to start the Docker container with GPU support:
```bash
sudo docker run -it --rm \
  --network host \
  --name openfl \
  --runtime=nvidia \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  320158lcc/openfl-611:latest \
  bash
```

### Initialize and Start the Framework

1. On the OpenFL server, initialize the model:
   ```bash
   fx plan initialize
   ```
   This creates the initial model file (`init.pbuf`).

2. Start the aggregator on the server:
   ```bash
   fx aggregator start
   ```

3. Start the collaborators on the client machines:
   - On client 1:
     ```bash
     fx collaborator start -n 1
     ```
   - On client 2:
     ```bash
     fx collaborator start -n 2
     ```

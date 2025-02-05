# OpenFL

A Framework for Federated Learning

## Prerequisites

Before running the framework, execute the following command to remove Docker RAM usage limitations:
```bash
ulimit -s unlimited
```

For more detailed information, please refer to the [official OpenFL documentation](https://openfl.readthedocs.io/en/latest/).

## Important Notes

1. **Environment Consistency**: 
   - Ensure all servers and clients have identical settings and code before starting the federated learning process
   - Any inconsistency in configurations or code may lead to training failures

2. **Code Updates**:
   - Whenever you modify the model code, you must re-run `fx plan initialize`
   - This ensures the initial model reflects your latest code changes

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

### Model Export

After the federated learning process completes:

1. Two model versions will be automatically saved in the `save` directory:
   - `last.pbuf`: The final model from the last training round
   - `best.pbuf`: The model that achieved the highest score during training

2. To export either model to SavedModel format on the server, use:
   ```bash
   # Export the final model
   fx model save -i save/last.pbuf

   # Or export the best performing model
   fx model save -i save/best.pbuf
   ```
   - On client 2:
     ```bash
     fx collaborator start -n 2
     ```

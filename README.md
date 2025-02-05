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

3. **Configuration Settings**:
   - You need to configure `plan/plan.yaml` according to your requirements
   - Key configuration sections include:
     - **Network**: Server address (`agg_addr`), port (`agg_port`), and TLS settings
     - **Training**: Number of epochs, batch size, and rounds to train
     - **Model Paths**: Locations for initial, best, and final model states
     - **Data Loading**: Batch size and data path settings
     - **Metrics**: Evaluation metrics and validation settings
     - **Aggregation**: How model updates are combined across clients

   Example of key settings in `plan.yaml`:
   ```yaml
   aggregator:
     settings:
       rounds_to_train: 5          # Total training rounds
       best_state_path: save/best.pbuf
       last_state_path: save/last.pbuf
   
   network:
     settings:
       agg_addr: 192.168.1.24     # Server address
       agg_port: 8081             # Server port
   
   data_loader:
     settings:
       batch_size: 128
       collaborator_count: 2
   
   tasks:
     train:
       kwargs:
         epochs: 10               # Epochs per round
         batch_size: 128
   ```

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

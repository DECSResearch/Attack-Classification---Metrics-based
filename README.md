# Openfl
## A federated learning framework

#ulimit -s unlimited

([reference](https://openfl.readthedocs.io/en/latest/))

- docker run -it --rm --network host --name openflserver --rm -e GRPC_VERBOSITY=DEBUG -e GRPC_TRACE=all changcunlei/openfl-test-tensorflow2.14-l4t-r36.2.0-bilstm:latest bash
- docker run -it --rm --network host --name openflclient1 --rm -e GRPC_VERBOSITY=DEBUG -e GRPC_TRACE=all changcunlei/openfl-test-tensorflow2.14-l4t-r36.2.0-bilstm:latest bash
- docker run -it --rm --network host --name openflclient2 --rm -e GRPC_VERBOSITY=DEBUG -e GRPC_TRACE=all changcunlei/openfl-test-tensorflow2.14-l4t-r36.2.0-bilstm:latest bash

- docker run -it --rm --network host --name openflserver --rm --runtime=nvidia --gpus all -e NVIDIA_VISIBLE_DEVICES=all changcunlei/openfl-test-tensorflow2.14-l4t-r36.2.0-bilstm:latest bash
- docker run -it --rm --network host --name openflclient1 --rm --runtime=nvidia --gpus all -e NVIDIA_VISIBLE_DEVICES=all changcunlei/openfl-test-tensorflow2.14-l4t-r36.2.0-bilstm:latest bash
- docker run -it --rm --network host --name openflclient2 --rm --runtime=nvidia --gpus all -e NVIDIA_VISIBLE_DEVICES=all changcunlei/openfl-test-tensorflow2.14-l4t-r36.2.0-bilstm:latest bash

- In openflserver run "fx aggregator start"
- In openflclient1 run "fx collaborator start -n 1"
- In openflclient2 run "fx collaborator start -n 2"

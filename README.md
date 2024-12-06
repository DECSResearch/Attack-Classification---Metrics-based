using bash script

you may need to run "sed -i 's/\r$//' stressing.sh" before running the script

Still need to use docker, otherwise you have to set up the whole tensorflow and cuda environment on the nano

sudo docker run -it --rm --network host --name stressing --rm --runtime=nvidia -v /usr/bin/tegrastats:/usr/bin/tegrastats --gpus all -e NVIDIA_VISIBLE_DEVICES=all changcunlei/attack-simulation:latest bash

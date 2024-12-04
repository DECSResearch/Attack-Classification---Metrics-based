FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

ENV PYTHONUNBUFFERED=1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_GPU_ALLOCATOR=cuda_malloc_async
ENV OMP_NUM_THREADS=1
ENV DEBIAN_FRONTEND=noninteractive

RUN python3 -m pip install --upgrade pip

WORKDIR /app

COPY . /app/

EXPOSE 8080

CMD ["/bin/bash"]

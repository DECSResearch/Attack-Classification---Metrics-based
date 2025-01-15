# 使用支持Jetson的TensorFlow镜像（ARM64架构）

#FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3
FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

#FROM python:3.9-slim

# 设置Python不缓冲输出，确保实时日志
ENV PYTHONUNBUFFERED=1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
RUN python3 -m pip install --upgrade pip
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_GPU_ALLOCATOR=cuda_malloc_async
ENV OMP_NUM_THREADS=1
# 防止apt-get安装时的交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 设置工作目录
WORKDIR /app

# 更新系统并安装必要的依赖
RUN apt-get update && apt-get install -y \
    nano \
    pkg-config \
    libhdf5-dev \
    gcc \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 升级pip
RUN python3 -m pip install --upgrade pip

# 卸载 PyYAML 解决方案：使用 apt-get 和 pip 卸载
#RUN apt-get remove -y python3-yaml && \
#    pip3 uninstall -y PyYAML || true && \
#    pip3 install --no-cache-dir numpy==1.24.3 && \
#    pip3 install --no-cache-dir pandas==2.0.3 && \
#    pip3 install --no-cache-dir scikit-learn && \
#    pip3 install --no-cache-dir openfl==1.6 && \
#    pip3 install --no-cache-dir requests==2.32.3
RUN pip3 install --no-deps openfl && \
    pip3 install --no-deps click && \
    pip3 install --no-deps rich && \
    pip3 install --no-deps dynaconf && \
    pip3 install --no-deps tqdm && \
    pip3 install --no-deps tensorboardx


# 复制项目文件
COPY . /app/

# 开放端口
EXPOSE 8080

# 使用bash作为默认命令
CMD ["/bin/bash"]
####################################(With GPU)############################################################

FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3

ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the source code
COPY FL_client/src/ /app/src

# Set the working directory to /app/src/
WORKDIR /app/src

# Install necessary packages and pip
RUN apt-get update && apt-get install -y \
    nano \
    pkg-config \
    libhdf5-dev \
    gcc \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

#RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Change working directory back to /app/
WORKDIR /app

# Keep the container running (replace with your actual command)
CMD ["tail", "-f", "/dev/null"]
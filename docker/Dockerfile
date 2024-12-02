FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

ENV PYTHONUNBUFFERED=1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV TF_FORCE_GPU_ALLOW_GROWTH=true
ENV TF_GPU_ALLOCATOR=cuda_malloc_async
ENV OMP_NUM_THREADS=1
ENV DEBIAN_FRONTEND=noninteractive

# Skip libc-bin triggers
RUN echo "exit 0" > /usr/sbin/policy-rc.d && \
    chmod +x /usr/sbin/policy-rc.d

# Prevent dpkg from running triggers
RUN echo '#!/bin/sh\nexit 0' > /usr/sbin/ldconfig && \
    chmod +x /usr/sbin/ldconfig

# Download and install network tools
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get download \
    tcpdump \
    nmap \
    wireshark \
    wireshark-qt \
    tshark \
    netcat-openbsd \
    && dpkg --force-not-root --force-depends -i *.deb || true && \
    rm *.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Fix QT display for Wireshark
ENV QT_X11_NO_MITSHM=1
ENV DISPLAY=:0

# Update pip
RUN python3 -m pip install --upgrade pip

WORKDIR /app

COPY . /app/

EXPOSE 8080

# Add non-root user and group for wireshark
RUN groupadd wireshark && \
    useradd -m -s /bin/bash netuser && \
    usermod -aG wireshark netuser

USER netuser

CMD ["/bin/bash"]
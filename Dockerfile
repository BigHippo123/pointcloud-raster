# PCR Development Docker Image
# Multi-stage build for minimal runtime image with GPU support.
# Usage: docker build -t pcr . && docker run -it pcr
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    libgdal-dev \
    libproj-dev \
    libomp-dev \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Build PCR
WORKDIR /build
COPY . .
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DPCR_ENABLE_CUDA=ON -DBUILD_PYTHON=ON && \
    cmake --build . -j$(nproc) && \
    cmake --install .

# Runtime image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    libgdal30 \
    libproj22 \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy built library
COPY --from=builder /usr/local /usr/local
COPY --from=builder /build/python/pcr /usr/local/lib/python3.10/dist-packages/pcr

# Install Python dependencies
RUN pip3 install numpy

WORKDIR /workspace
CMD ["python3"]

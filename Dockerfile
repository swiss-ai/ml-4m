# Boilerplate Swiss AI stuff
FROM nvcr.io/nvidia/pytorch:24.01-py3
 
ENV DEBIAN_FRONTEND=noninteractive
ENV http_proxy=http://proxy.cscs.ch:8080
ENV https_proxy=https://proxy.cscs.ch:8080

# Install basic commands 
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get update && apt-get install -y \
python3-pip \
python3-venv \
git
WORKDIR /workspace
COPY ml-4m /workspace/ml-4m
WORKDIR /workspace/ml-4m
RUN pip install -e .
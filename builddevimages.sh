#!/bin/bash

# Navigate to the directory containing your Dockerfiles if it's not the current directory
# cd path/to/dockerfiles

# Build the base CUDA image
echo "Building tolleybot/cuda..."
docker build -f docker/Dockerfile.cuda -t tolleybot/cuda .

# Build the vcpkg image, which depends on tolleybot/cuda
echo "Building tolleybot/vcpkg..."
docker build -f docker/Dockerfile.vcpkg -t tolleybot/vcpkg .

# Build the ONNX image, which depends on tolleybot/vcpkg
echo "Building tolleybot/onnx..."
docker build -f docker/Dockerfile.onnxdev -t tolleybot/onnx .

# Build the ONNX Runtime image, which depends on tolleybot/onnx
echo "Building tolleybot/onnxruntime..."
docker build -f docker/Dockerfile.onnxruntime -t tolleybot/onnxruntime .

# Build the ONNX Extensions image, which depends on tolleybot/onnxruntime
echo "Building tolleybot/onnxext..."
docker build -f docker/Dockerfile.onnxext -t tolleybot/onnxext .

docker tag tolleybot/onnxext tolleybot/onnxdev:latest

echo "Build process completed!"

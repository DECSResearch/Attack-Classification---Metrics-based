#!/bin/bash

# Set variables
IMAGE_NAME="changcunlei/openfl-test-tensorflow2.14-l4t-r36.2.0-bilstm"
TAG="latest"

echo "====== Starting Docker Build and Deployment Process ======"

# 1. Stop and remove existing containers (if they exist)
echo "1. Cleaning up existing containers"
docker-compose down

# 2. Build Docker image (add --platform to specify architecture)
echo "2. Building Docker image"
docker build --platform linux/arm64 -t $IMAGE_NAME:$TAG -f ./docker/Dockerfile .

# Check build result
if [ $? -eq 0 ]; then
    echo "✅ Docker image build successful"
else
    echo "❌ Docker image build failed"
    exit 1
fi

# 3. Login to Docker Hub
echo "3. Logging in to Docker Hub"
docker login

# 4. Push image to Docker Hub
echo "4. Pushing image to Docker Hub"
docker push $IMAGE_NAME:$TAG

# Check push result
if [ $? -eq 0 ]; then
    echo "✅ Image successfully pushed to Docker Hub"
else
    echo "❌ Push failed"
    exit 1
fi

# 5. Start services using docker-compose
echo "5. Starting services"
docker-compose up -d

# 6. Check service status
echo "6. Checking service status"
docker-compose ps

# 7. Display network information
echo "7. Displaying network information"
docker network ls | grep federated-net

echo "====== Complete ======"
echo "Image uploaded to: $IMAGE_NAME:$TAG"
echo "Container started, you can view logs using:"
echo "docker-compose logs -f"
echo "You can enter the container using:"
echo "docker exec -it openfl-client-1 bash"
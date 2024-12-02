#!/bin/bash

IMAGE_NAME="changcunlei/attack-simulation"
TAG="latest"

echo "====== 开始Docker构建和发布流程 ======"

# 1. 停止并删除现有容器（如果存在）
echo "1. 清理现有容器"
docker-compose down

# 2. 构建Docker镜像（添加 --platform 指定架构）
echo "2. 构建Docker镜像"
docker build --platform linux/arm64 -t $IMAGE_NAME:$TAG -f ./docker/Dockerfile .

# 检查构建结果
if [ $? -eq 0 ]; then
    echo "✅ Docker镜像构建成功"
else
    echo "❌ Docker镜像构建失败"
    exit 1
fi

# 3. 登录到Docker Hub
echo "3. 登录到Docker Hub"
docker login

# 4. 推送镜像到Docker Hub
echo "4. 推送镜像到Docker Hub"
docker push $IMAGE_NAME:$TAG

# 检查推送结果
if [ $? -eq 0 ]; then
    echo "✅ 镜像已成功推送到Docker Hub"
else
    echo "❌ 推送失败"
    exit 1
fi

# 5. 使用docker-compose启动服务
echo "5. 启动服务"
docker-compose up -d

# 6. 检查服务状态
echo "6. 检查服务状态"
docker-compose ps

# 7. 显示网络信息
echo "7. 显示网络信息"
docker network ls | grep federated-net

echo "====== 完成 ======"
echo "镜像已上传至: $IMAGE_NAME:$TAG"
echo "容器已启动，可以使用以下命令查看日志："
echo "docker-compose logs -f"
echo "可以使用以下命令进入容器："
echo "docker exec -it openfl-client-1 bash"

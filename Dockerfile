# 使用官方Python 3.9镜像作为基础镜像
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.9-slim-linuxarm64

COPY . /app
# 创建工作目录
WORKDIR /app

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.doubanio.com/simple/

# 暴露5000端口
EXPOSE 5000

# 设置启动命令
CMD ["python", "main.py"]
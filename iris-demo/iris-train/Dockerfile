# 1. 改用完整版 Python 镜像（解决缺编译器的问题）
FROM python:3.9

WORKDIR /app

COPY requirements.txt .
COPY train.py .

# 2. 先升级 pip（解决找不到包的问题）
RUN pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/

# 3. 改用阿里云源安装依赖（这时候清华源可能不稳定）
# 增加了 --trusted-host 参数，防止 SSL 报错
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

CMD ["python", "train.py"]
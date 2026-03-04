FROM python:3.9

WORKDIR /app

COPY requirements.txt .
COPY server.py .

RUN pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/

RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

# cube-studio 要求的默认端口通常是 80 或 8080
EXPOSE 8080

ENTRYPOINT ["python", "server.py"]
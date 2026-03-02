FROM python:3.9

WORKDIR /app

COPY requirements.txt .
COPY train.py .

RUN pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/

RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

ENTRYPOINT ["python", "train.py"]

FROM python:3.9

WORKDIR /app

COPY requirements.txt .
COPY evaluate.py .
COPY evaluate_classification.py .
COPY evaluate_regression.py .

RUN pip install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple/

RUN pip install --no-cache-dir -r requirements.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

ENTRYPOINT ["python", "evaluate.py"]

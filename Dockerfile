FROM python:3.10-slim AS requirements 

WORKDIR /tmp

RUN pip install --upgrade pip && \
    pip install dvc[s3]

COPY models/model.onnx.dvc ./models/model.onnx.dvc

RUN --mount=type=secret,id=aws_access_key,required \
    --mount=type=secret,id=aws_secret_key,required \
    sh -c ' \
    export AWS_ACCESS_KEY_ID=$(cat /run/secrets/aws_access_key) && \
    export AWS_SECRET_ACCESS_KEY=$(cat /run/secrets/aws_secret_key) && \
    dvc init --no-scm && \
    dvc remote add -d storage s3://customerintelligence/advanced-analytics/ssafarveisi/mlops && \
    dvc remote modify storage endpointurl http://s3-de-central.profitbricks.com && \
    dvc remote modify storage region eu-central-1 && \
    dvc pull models/model.onnx.dvc \
    '

FROM huggingface/transformers-pytorch-cpu:latest

COPY --from=requirements /tmp/models/model.onnx ./models/model.onnx
COPY ./requirements_inference.txt app.py data.py inference_onnx.py ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_inference.txt 

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
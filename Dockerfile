FROM python:3.11

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y && apt-get clean
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
COPY . /app
EXPOSE 5000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "5000"]
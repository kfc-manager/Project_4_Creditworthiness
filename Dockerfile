FROM python:3.10.0-slim

WORKDIR /app

COPY ./requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY ./kredit.dat ./kredit.dat

COPY ./main.py ./main.py

CMD ["python3", "main.py"]

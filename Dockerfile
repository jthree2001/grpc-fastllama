FROM python:3.10

WORKDIR /srv
COPY ./requirements.txt .

RUN python3 -m venv venv && . venv/bin/activate
RUN python3 -m pip install --no-cache-dir -r requirements.txt --upgrade pip

COPY ./main.py /srv/main.py
COPY ./config.py /srv/config.py
COPY ./chats.py /srv/chats.py
COPY ./protos /srv/protos
COPY ./chat_pb2.py /srv/chat_pb2.py
COPY ./custom /srv/custom

COPY ./streamer /srv/streamer
# COPY ./models /srv/models  # Mounting model is more efficient
# CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "9600", "--db_path", "data/database.db"]

# python server.py --auto-devices --api --listen
CMD ["python", "main.py"]
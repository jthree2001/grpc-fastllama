FROM jthree2001/gpt4all-rpc:alpha-worker

WORKDIR /srv
COPY ./requirements.txt .

RUN python3 -m venv venv && . venv/bin/activate
RUN python3 -m pip install --no-cache-dir -r requirements.txt --upgrade pip

COPY ./main.py /srv/main.py
COPY ./config.py /srv/config.py
COPY ./chats.py /srv/chats.py
COPY ./protos /srv/protos

# COPY ./models /srv/models  # Mounting model is more efficient
# CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "9600", "--db_path", "data/database.db"]
CMD ["python", "main.py"]
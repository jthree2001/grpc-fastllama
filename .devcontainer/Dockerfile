FROM python:3.10

# WORKDIR /srv
COPY ./requirements.txt .

RUN python3 -m venv venv && . venv/bin/activate
RUN python3 -m pip install --no-cache-dir -r requirements.txt --upgrade pip

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install --no-install-recommends --assume-yes \
      protobuf-compiler
# COPY ./models /srv/models  # Mounting model is more efficient
# CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "9600"]

# python gpt4all_grpc.py --db_path data/database.db
# python app.py --host 0.0.0.0 --port 9600 --db_path data/database.db
# python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. protos/chat.proto
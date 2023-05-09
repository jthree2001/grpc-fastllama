# grpc-fastllama
Simple Api interface using LangChain


Generating grpc code:

python -m grpc_tools.protoc --proto_path=protos chat.proto --python_out=. --grpc_python_out=protos
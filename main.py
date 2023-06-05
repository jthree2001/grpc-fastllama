import grpc
import argparse
from config import load_config
from concurrent import futures
import chat_pb2 as chat_pb2
import protos.chat_pb2_grpc as chat_pb2_grpc
from google.protobuf import json_format
from redis import Redis
from rq import Queue
from chats import ChatbotInstance

class FastLLamaGRPC(chat_pb2_grpc.ChatServiceServicer):

    def __init__(self, config:dict) -> None:
        super 
        self.config = config
        self.queue = Queue(connection=Redis(host=config["redis_host"], port=config["redis_port"], db=config["redis_db"]))

    def SendChatMessage(self, request, context):
        print("Reciving chat message")
        enqueue_message = self.queue.enqueue(ChatbotInstance.generate_in_worker, request.chat_id, self.config, request.callback_url, request.message, job_timeout=self.config["job_timeout"])
        print(enqueue_message)
        print("Enqueued chat message")
        reply = "Results will be sent async"
        return chat_pb2.SendChatMessageResponse(message=reply)

    def CreateChat(self, request, context):
        print("Create new chat")
        new_chat = ChatbotInstance(self.db, self.config)
        new_chat.prepare_a_new_chatbot()
        return chat_pb2.CreateChatResponse(chat=chat_pb2.Chat(id=new_chat.id(), title=new_chat.title()))

    def SendChainMessage(self, request, context):
        print("Reciving chat message")
        enqueue_message = self.queue.enqueue(ChatbotInstance.chain_in_worker, request.chat_id, self.config, request.callback_url, request.message, job_timeout=self.config["job_timeout"])
        print(enqueue_message)
        print("Enqueued chat message")
        reply = "Results will be sent async"
        return chat_pb2.SendChatMessageResponse(message=reply)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Start the chatbot Flask app.")
    parser.add_argument(
        "-s", "--seed", type=int, default=None, help="Force using a specific model."
    )

    parser.add_argument(
        "-m", "--model", type=str, default=None, help="Force using a specific model."
    )
    parser.add_argument(
        "-em", "--embedd_model", type=str, default=None, help="Force using a specific model for embedded LLM worker."
    )
    parser.add_argument(
        "--temp", type=float, default=None, help="Temperature parameter for the model."
    )
    parser.add_argument(
        "--n_predict",
        type=int,
        default=None,
        help="Number of tokens to predict at each step.",
    )
    parser.add_argument(
        "--top_k", type=int, default=None, help="Value for the top-k sampling."
    )
    parser.add_argument(
        "--top_p", type=float, default=None, help="Value for the top-p sampling."
    )
    parser.add_argument(
        "--repeat_penalty", type=float, default=None, help="Penalty for repeated tokens."
    )
    parser.add_argument(
        "--repeat_last_n",
        type=int,
        default=None,
        help="Number of previous tokens to consider for the repeat penalty.",
    )
    parser.add_argument(
        "--ctx_size",
        type=int,
        default=None,#2048,
        help="Size of the context window for the model.",
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="the hostname to listen on"
    )
    parser.add_argument("--port", type=int, default=None, help="the port to listen on")
    parser.add_argument("--threads", type=int, default=None, help="How many threads to use")
    parser.add_argument("--redis_host", type=str, default="192.168.254.85", help="Redis Host for rq")
    parser.add_argument("--redis_port", type=int, default=6379, help="Redis Port for rq")
    parser.add_argument("--redis_db", type=int, default=1, help="Redis Database for data separation")
    parser.add_argument("--job_timeout", type=int, default=1800, help="Worker Timeout")
    parser.set_defaults(debug=False)
    args = parser.parse_args()
    config_file_path = "configs/default.yaml"
    config = load_config(config_file_path)

    # Override values in config with command-line arguments
    for arg_name, arg_value in vars(args).items():
        if arg_value is not None:
            config[arg_name] = arg_value
    

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    chat_pb2_grpc.add_ChatServiceServicer_to_server(
        FastLLamaGRPC(config), server)
    server.add_insecure_port('[::]:'+str(config["port"]))
    server.start()
    print("Ready...")
    print(config)
    server.wait_for_termination()
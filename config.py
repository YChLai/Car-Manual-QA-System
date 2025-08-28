import os

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
LLM_MODEL_NAME = "glm-4"
LLM_TEMPERATURE = 0.1

VECTORSTORE_PATH = "../Car-Manual/chroma_db"
PARENT_STORE_PATH = "./parent_store_data.pkl"

NUM_QUERIES_TO_GENERATE = 4 
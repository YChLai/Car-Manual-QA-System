import pickle
import warnings
import logging
import traceback

from langchain_community.chat_models import ChatZhipuAI
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.stores import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings

import config
import chains

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def main():
    """主函数，负责初始化和运行聊天机器人"""

    print(">>> 正在初始化核心组件...")

    llm = ChatZhipuAI(model=config.LLM_MODEL_NAME, api_key=config.ZHIPU_API_KEY, temperature=config.LLM_TEMPERATURE)
    embedding = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

    print(">>> 正在从本地加载数据库...")
    child_vectorstore = Chroma(
        persist_directory=config.VECTORSTORE_PATH,
        embedding_function=embedding
    )

    try:
        with open(config.PARENT_STORE_PATH, "rb") as f:
            parent_chunks, doc_ids = pickle.load(f)
        parent_store = InMemoryStore()
        parent_store.mset(list(zip(doc_ids, parent_chunks)))
    except FileNotFoundError:
        print(f"错误: 找不到父文档存储文件 {config.PARENT_STORE_PATH}")
        return

    print(">>> 数据库和存储库加载完成。")

    print(">>> 正在构建 RAG 链...")
    rag_chain = chains.create_rag_fusion_chain(llm, child_vectorstore, parent_store)
    print(">>> 请开始提问：")

    chat_history = []
    while True:
        try:
            query = input("\n用户问题 (输入 'exit' 退出): ")
            if query.lower() == "exit":
                print("再见！")
                break
            if not query:
                continue

            inputs = {
                "question": query,
                "chat_history": chat_history,
                "num_queries": config.NUM_QUERIES_TO_GENERATE
            }

            print("--- AI 思考中 ... ---")
            answer = rag_chain.invoke(inputs)

            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=answer))
            chat_history = chat_history[-10:]  # 保持最近5轮对话

            print("\n--- AI 回答 ---")
            print(answer)

        except Exception as e:
            print(f"\n发生错误: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
import os
import pickle
import warnings

from langchain_community.chat_models import ChatZhipuAI
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.stores import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

print(">>> 正在初始化核心组件...")

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
VECTORSTORE_PATH = "./chroma_db"
PARENT_STORE_PATH = "./parent_store_data.pkl"

llm = ChatZhipuAI(model="glm-4", api_key=ZHIPU_API_KEY)
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

print(">>> 正在从本地加载数据库...")

child_vectorstore = Chroma(
    persist_directory=VECTORSTORE_PATH,
    embedding_function=embedding
)
with open(PARENT_STORE_PATH, "rb") as f:
    parent_chunks, doc_ids = pickle.load(f)
parent_store = InMemoryStore()
parent_store.mset(list(zip(doc_ids, parent_chunks)))

print(">>> 数据库和存储库加载完成。")

query_decomposition_prompt = ChatPromptTemplate.from_template(
    """
        你是一个专门用于RAG系统的查询优化专家。
    你的任务是将一个复杂或模糊的用户问题，分解成一个或多个具体的、可以独立进行搜索的子问题列表。
    你必须以一个包含 "queries" 键的JSON对象格式返回结果，该键的值是一个字符串列表。
    
    ---
    示例 1:
    用户问题: "model 3和model y的钥匙类型有区别吗"
    你的回答: 
    {{
      "queries": [
        "model 3的钥匙类型有哪些？如何使用？",
        "model y的钥匙类型有哪些？如何使用？"
      ]
    }}
    ---
    示例 2:
    用户问题: "介绍一下车辆的安全性能"
    你的回答:
    {{
      "queries": [
        "车辆有哪些主动安全功能？",
        "车辆有哪些被动安全功能，比如安全气气囊？",
        "车辆的官方碰撞测试评级是多少？"
      ]
    }}
    ---
    现在，请处理以下问题：
    用户问题: "{query}"
    你的回答:
    """
)

query_decomposition_chain = (
    query_decomposition_prompt
    | llm
    | JsonOutputParser()
    | (lambda x: x.get("queries", []))
)

document_content_description = "汽车的用户手册，包含了不同型号汽车的功能介绍和使用说明"
metadata_field_info = [
    AttributeInfo(
        name="source",
        description='文档的来源文件名，代表了汽车的型号。可选值为 ["./model_3.pdf", "./model_y.pdf", "./model_s.pdf"]',
        type="string",
    )
]
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=child_vectorstore,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    verbose=True,
)

def get_parent_docs(child_docs):
    parent_ids = {doc.metadata.get("parent_id") for doc in child_docs}
    if not parent_ids:
        return []
    return parent_store.mget(list(parent_ids))

retriever_chain = self_query_retriever | RunnableLambda(get_parent_docs)

def flatten_and_deduplicate_docs(list_of_lists):
    all_docs = []
    for sublist in list_of_lists:
        for doc in sublist:
            if doc:
                all_docs.append(doc)

    unique_docs_dict = {doc.page_content: doc for doc in all_docs}

    return list(unique_docs_dict.values())

context_chain = (
    query_decomposition_chain
    | retriever_chain.map()
    | RunnableLambda(flatten_and_deduplicate_docs)
)

def format_docs(docs):
    if not docs:
        return "未找到相关上下文信息。"
    sources = {os.path.basename(doc.metadata.get("source", "未知来源")) for doc in docs if doc}
    content = "\n\n".join(doc.page_content for doc in docs if doc)
    return f"{content}\n\n来源: {', '.join(sources)}"

final_prompt = ChatPromptTemplate.from_template(
    """
    根据以下检索到的上下文信息，用中文简洁地回答用户的问题。
    如果上下文中没有相关信息，就直接说“根据现有文档，我无法回答这个问题”。
    不要编造答案。在回答的末尾，请另起一行，用'来源: [文件名]'的格式，注明信息来自哪个文档。

    上下文: 
    {context}

    问题: 
    {question}
    """
)

# 最终的RAG链
rag_chain = (
    {"context": {"query": RunnablePassthrough()} | context_chain | RunnableLambda(format_docs),
     "question": RunnablePassthrough()}
    | final_prompt
    | llm
    | StrOutputParser()
)

print("\n--- 系统准备就绪！---\n")
query = "Model Y、3、S的辅助驾驶有什么区别？"
print(f"用户问题: {query}")

answer = rag_chain.invoke(query)
print("\n--- AI回答 ---\n")
print(answer)


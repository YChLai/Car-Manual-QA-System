import os
import pickle
import warnings
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.stores import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo

# --- 0. 初始设置 ---
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. 初始化核心组件 (一次性完成) ---
print(">>> 正在初始化核心组件...")
llm = ChatZhipuAI(model="glm-4", api_key=os.getenv("ZHIPU_API_KEY"))
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# --- 2. 加载持久化数据 ---
print(">>> 正在从本地加载数据库...")
vectorstore_path = "./chroma_db"
# 加载Chroma数据库
child_vectorstore = Chroma(
    persist_directory=vectorstore_path,
    embedding_function=embedding
)
with open("./parent_store_data.pkl", "rb") as f:
    parent_chunks, doc_ids = pickle.load(f)
parent_store = InMemoryStore()
parent_store.mset(list(zip(doc_ids, parent_chunks)))
print(">>> 数据库和存储库加载完成。")

# --- 3. 创建高级检索器 (一次性完成) ---
print(">>> 正在创建高级检索器...")
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
    verbose=True, # 开启详细模式，方便调试
)

# 创建一个函数，用于根据子文档取回父文档
def get_parent_docs(child_docs):
    print("--- 正在根据子文档取回父文档 ---")
    parent_ids = [doc.metadata.get("parent_id") for doc in child_docs]
    unique_parent_ids = list(set(parent_ids))
    return parent_store.mget(unique_parent_ids)

# 将自查询检索器和父文档检索逻辑连接成一个检索链
retriever_chain = self_query_retriever | RunnableLambda(get_parent_docs)

# --- 4. 构建并运行RAG链 ---
print(">>> 正在构建并运行RAG问答链...")
template = """
根据以下检索到的上下文信息，用中文简洁地回答用户的问题。
如果上下文中没有相关信息，就直接说“根据现有文档，我无法回答这个问题”。
不要编造答案。在回答的末尾，请另起一行，用'来源: [文件名]'的格式，注明信息来自哪个文档。

上下文: 
{context}

问题: 
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    if not docs:
        return "未找到相关上下文信息。"
    content = "\n\n".join(doc.page_content for doc in docs if doc)
    sources = ", ".join(list(set(os.path.basename(doc.metadata.get("source", "未知来源")) for doc in docs if doc)))
    return f"{content}\n\n来源: {sources}"

# 构建最终的RAG链
rag_chain = (
    {"context": retriever_chain | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 5. 提问 ---
print("--- 系统准备就绪！---\n")
query = ("Model S座椅安全带注意事项有哪些？")
print(f"用户问题: {query}")
answer = rag_chain.invoke(query)
print("\n--- AI回答 ---\n")
print(answer)
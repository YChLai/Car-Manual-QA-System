import pickle
import os
from dotenv import load_dotenv
import warnings

# --- 导入 LangChain 相关库 ---
from langchain_core.stores import InMemoryStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 仅用于重建retriever时的类型提示
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatZhipuAI

# --- 屏蔽不必要的警告 ---
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. 初始化和加载 ---

print(">>> 正在初始化组件...")

# 初始化GLM-4模型 (使用LangChain的官方集成)
llm = ChatZhipuAI(
    model="glm-4",
    api_key=os.environ.get("ZHIPU_API_KEY")
)

# 初始化嵌入模型
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

# 从本地加载数据库
print(">>> 正在从本地加载数据库...")
vectorstore_path = "./faiss_index"
child_vectorstore = FAISS.load_local(
    vectorstore_path,
    embedding,
    allow_dangerous_deserialization=True
)

with open("./parent_store_data.pkl", "rb") as f:
    parent_chunks, doc_ids = pickle.load(f)

parent_store = InMemoryStore()
parent_store.mset(list(zip(doc_ids, parent_chunks)))

print(">>> 数据库和存储库加载完成。")


# --- 2. 自定义高级检索逻辑 ---

def custom_retriever(query: str):
    """
    自定义的父文档检索逻辑：
    1. 在子文档向量库中进行相似度搜索。
    2. 提取子文档对应的父文档ID。
    3. 从父文档库中获取并返回去重后的父文档。
    """
    print(f"--- 正在对问题进行自定义检索: '{query}' ---")

    # 1. 直接在子文档库中搜索
    similar_child_docs = child_vectorstore.similarity_search(query, k=5)

    if not similar_child_docs:
        print("--- 未找到相似的子文档 ---")
        return []

    # 2. 从子文档中提取出父文档的ID
    parent_ids = [doc.metadata.get("parent_id") for doc in similar_child_docs]
    unique_parent_ids = list(set(parent_ids))

    # 3. 从父文档库中取回完整的父文档
    retrieved_parents = parent_store.mget(unique_parent_ids)
    final_docs = [doc for doc in retrieved_parents if doc]  # 过滤掉可能存在的None值

    print(f"--- 成功检索到 {len(final_docs)} 份相关的父文档 ---")
    return final_docs


# --- 3. 构建RAG链 ---

print(">>> 正在构建RAG问答链...")

# 定义提示词模板
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


# 定义一个函数来格式化检索到的文档
def format_docs(docs):
    if not docs:
        return "未找到相关上下文信息。"
    content = "\n\n".join(doc.page_content for doc in docs)
    sources = ", ".join(list(set(os.path.basename(doc.metadata.get("source", "未知来源")) for doc in docs)))
    return f"{content}\n\n来源: {sources}"


# 构建链，使用我们自己的 custom_retriever
rag_chain = (
        {"context": custom_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

# --- 4. 运行问答 ---

print("--- 系统准备就绪！---\n")

query = "Model S如何自动开启后备箱？"
print(f"用户问题: {query}")

# 调用链来获取答案
answer = rag_chain.invoke(query)

print("\n--- AI回答 ---\n")
print(answer)
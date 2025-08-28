import os
import pickle
import warnings
import logging
import traceback

from langchain.chains.query_constructor.base import load_query_constructor_runnable
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.chains.query_constructor.base import Comparator, Operator

from langchain_community.chat_models import ChatZhipuAI
from langchain_community.vectorstores import Chroma
from langchain_community.query_constructors.chroma import ChromaTranslator

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.stores import InMemoryStore

from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


print(">>> 正在初始化核心组件...")

ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
VECTORSTORE_PATH = "../Car-Manual/chroma_db"
PARENT_STORE_PATH = "./parent_store_data.pkl"
NUM_QUERIES_TO_GENERATE = 6

llm = ChatZhipuAI(model="glm-4", api_key=ZHIPU_API_KEY, temperature=0.1)
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


# --- 2. 自查询检索器 (Self-Query Retriever) 定义 ---
print(">>> 正在构建 RAG 链...")

document_content_description = (
    "这是一系列关于不同型号汽车的用户手册。"
    "你只能使用 'source' 这一个元数据字段进行过滤。"
    "绝对不允许编造或使用 'source' 以外的任何其他字段进行过滤。"
)
metadata_field_info = [
    AttributeInfo(
        name="source",
        description=(
            '文档的来源文件名，代表了汽车的型号。'
            '当只有一个型号时，请使用 `eq` 操作符，例如：`eq("source", "./model_s.pdf")`。'
            '当需要查询多个型号时，才使用 `in` 操作符，并且值必须是列表，例如：`in("source", ["./model_3.pdf", "./model_x.pdf"])`。'
            '可选值为 ["./model_3.pdf", "./model_x.pdf", "./model_y.pdf", "./model_s.pdf"], 全为小写字母。'
        ),
        type="string",
    )
]

allowed_comparators = [
    Comparator.EQ, Comparator.NE, Comparator.GT, Comparator.GTE,
    Comparator.LT, Comparator.LTE, Comparator.IN,
]
allowed_operators = [Operator.AND, Operator.OR, Operator.NOT]

query_constructor_chain = load_query_constructor_runnable(
    llm=llm,
    document_contents=document_content_description,
    attribute_info=metadata_field_info,
    allowed_comparators=allowed_comparators,
    allowed_operators=allowed_operators,
)

structured_query_translator = ChromaTranslator()
structured_query_translator.allowed_comparators = allowed_comparators
structured_query_translator.allowed_operators = allowed_operators

self_query_retriever = SelfQueryRetriever(
    llm_chain=query_constructor_chain,
    vectorstore=child_vectorstore,
    structured_query_translator=structured_query_translator,
    verbose=True,
)

query_generation_template = """
你是一位顶级的AI助手，专注于RAG检索增强和查询优化。
你的目标是帮助用户克服单一、模糊查询的局限性，通过生成 {num_queries} 个优化后的查询来提升向量检索的召回率和准确性。

你将根据原始问题的性质，灵活运用以下三种策略之一或组合：

1.  **视角重写 (Perspective Rewrite)**: 保持问题的核心意图不变，但从不同角度或用不同措辞重新提问。
2.  **维度分解 (Aspect-based Decomposition)**: 将一个宽泛的、关于某个主题的问题，分解成关于该主题不同方面（如性能、价格、功能）的具体问题。
3.  **实体分解 (Entity-based Decomposition)**: 将一个需要对比多个实体的问题，分解成针对每个实体的独立问题。

---
### 示例 1: 视角重写 (Perspective Rewrite)
**原始问题**: "Model 3 和 Model Y 的辅助驾驶有区别吗？"
**生成的查询**:
特斯拉 Model 3 的 Autopilot 功能
特斯拉 Model Y 的辅助驾驶系统
Model 3 与 Model Y 在自动驾驶硬件上的差异
比较 Model 3 和 Model Y 的 FSD 软件升级路径

---
### 示例 2: 维度分解 (Aspect-based Decomposition)
**原始问题**: "介绍一下Model 3"
**生成的查询**:
Model 3 的智能驾驶系统怎么样
Model 3 的动力性能和续航里程
Model 3 的售价和不同配置的差异
Model 3 的内部空间和设计特点

---
### 示例 3: 实体分解 (Entity-based Decomposition)
**原始问题**: "对比一下Model 3和Model S的辅助驾驶"
**生成的查询**:
特斯拉 Model 3 的辅助驾驶功能有哪些
特斯拉 Model S 的辅助驾驶功能有哪些
Model 3 的辅助驾驶硬件配置
Model S 的辅助驾驶硬件配置

---
现在，请根据以下原始问题，严格按照上述原则和示例风格，生成 {num_queries} 个优化查询。
每个查询占一行，不要带任何编号或前缀。

**原始问题**: "{original_query}"
**生成的查询**:
"""
query_generation_prompt = ChatPromptTemplate.from_template(query_generation_template)
generate_queries_chain = (
    query_generation_prompt
    | llm
    | StrOutputParser()
    | (lambda x: x.strip().split("\n"))
)

def reciprocal_rank_fusion(list_of_doc_lists: list[list[tuple[int, any]]], k=60):
    ranked_results = {}
    for doc_list in list_of_doc_lists:
        if not doc_list:
            continue
        for rank, doc in enumerate(doc_list):
            doc_content = doc.page_content
            if doc_content not in ranked_results:
                ranked_results[doc_content] = {"score": 0, "doc": doc}
            ranked_results[doc_content]["score"] += 1 / (k + rank + 1)
    sorted_results = sorted(ranked_results.values(), key=lambda x: x["score"], reverse=True)
    return [item["doc"] for item in sorted_results]


# --- 4. 辅助函数和最终 Prompt 定义 ---

## 关键修正 2: 采用分离的辅助函数和Prompt，确保来源精确
def get_parent_docs(child_docs):
    parent_ids = {doc.metadata.get("parent_id") for doc in child_docs if doc and doc.metadata}
    if not parent_ids:
        return []
    return parent_store.mget(list(parent_ids))

def format_content(docs):
    if not docs:
        return "未找到相关上下文信息。"
    return "\n\n".join(doc.page_content for doc in docs if doc)

def format_sources(docs):
    if not docs:
        return ""
    sources = {os.path.basename(doc.metadata.get("source", "未知来源")) for doc in docs if doc and doc.metadata}
    return ", ".join(sources)

final_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        """你是一个专业的汽车手册问答助手。请根据以下提供的上下文信息，用中文简洁、全面地回答用户的问题。
如果上下文中明确指出“未找到相关上下文信息”，请直接告知用户文档中没有相关内容。

上下文:
{context}
---
在你完成对用户问题的回答后，你必须严格按照以下格式和内容，另起一行来结束你的回答：
来源: {sources}"""
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("用户问题: {question}")
])


# --- 5. 构建并运行最终的 RAG 链 ---
base_retriever_chain = self_query_retriever | RunnableLambda(get_parent_docs)

rag_fusion_chain = (
    RunnablePassthrough.assign(
        generated_queries=(
            lambda x: {"original_query": x["question"], "num_queries": x["num_queries"]}
        ) | generate_queries_chain
    )
    | RunnablePassthrough.assign(
        retrieved_doc_lists=(lambda x: x["generated_queries"]) | base_retriever_chain.map()
    )
    | RunnablePassthrough.assign(
        fused_docs=(lambda x: x["retrieved_doc_lists"]) | RunnableLambda(reciprocal_rank_fusion)
    )
    | RunnablePassthrough.assign(
        context = (lambda x: x["fused_docs"]) | RunnableLambda(format_content),
        sources = (lambda x: x["fused_docs"]) | RunnableLambda(format_sources)
    )
    | final_prompt
    | llm
    | StrOutputParser()
)

print(">>> 请开始提问：")


# --- 6. 问答循环 ---
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
            "num_queries": NUM_QUERIES_TO_GENERATE
        }

        print("--- AI 思考中 ... ---")
        answer = rag_fusion_chain.invoke(inputs)

        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=answer))
        chat_history = chat_history[-10:]

        print("\n--- AI 回答 ---")
        print(answer)

    except Exception as e:
        print(f"\n发生错误: {e}")
        traceback.print_exc()
from langchain.chains.query_constructor.base import load_query_constructor_runnable
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.chains.query_constructor.base import Comparator, Operator

from langchain_community.query_constructors.chroma import ChromaTranslator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

import utils

REPHRASE_QUESTION_SYSTEM_PROMPT = """根据一段对话历史和用户提出的后续问题，
将后续问题改写成一个独立的、完整的、无需上下文就能理解的新问题。
如果后续问题本身已经是一个独立的问题，则无需改写，直接返回原问题即可。
"""

rephrase_question_prompt = ChatPromptTemplate.from_messages([
    ("system", REPHRASE_QUESTION_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

def create_rag_fusion_chain(llm, vectorstore, parent_store):
    """创建并返回完整的RAG Fusion链"""

    document_content_description = (
        "这是一系列关于不同型号汽车的用户手册。"
        "进行元数据过滤时，你**唯一**可以使用的字段是 'source'。"
        "这是一个****绝对的规则****，**严禁使用、创造或联想任何 'source' 以外的字段进行过滤**。"
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
        query_constructor=query_constructor_chain,
        vectorstore=vectorstore,
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

    final_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            """
            你是一个专业的汽车手册问答助手。请根据以下提供的上下文信息，用中文简洁、全面地回答用户的问题。
            如果上下文中明确指出“未找到相关上下文信息”，请直接告知用户文档中没有相关内容。
                
            上下文:
            {context}
            ---
            在你完成对用户问题的回答后，你必须严格按照以下格式和内容，另起一行来结束你的回答：
            来源: {sources}
            """
            ),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("用户问题: {question}")
    ])

    rephrase_question_chain = rephrase_question_prompt | llm | StrOutputParser()
    get_parent_docs_with_store = RunnableLambda(lambda docs: utils.get_parent_docs(docs, parent_store))
    base_retriever_chain = self_query_retriever | get_parent_docs_with_store

    rag_fusion_chain = (
            RunnablePassthrough.assign(
                standalone_question=rephrase_question_chain
            )

            | RunnablePassthrough.assign(
                generated_queries=(
                        (lambda x: {"original_query": x["standalone_question"], "num_queries": x["num_queries"]}) | generate_queries_chain
                )
            )

            | RunnablePassthrough.assign(
                retrieved_doc_lists=(
                        (lambda x: x["generated_queries"]) | base_retriever_chain.map()
                )
            )

            | RunnablePassthrough.assign(
                fused_docs=(
                        (lambda x: x["retrieved_doc_lists"]) | RunnableLambda(utils.reciprocal_rank_fusion)
                )
            )

            | RunnablePassthrough.assign(
                context=(lambda x: x["fused_docs"]) | RunnableLambda(utils.format_content),
                sources=(lambda x: x["fused_docs"]) | RunnableLambda(utils.format_sources),
            )

            | final_prompt

            | llm

            | StrOutputParser()
    )

    return rag_fusion_chain
import os
from typing import List
from langchain_core.documents import Document


def get_parent_docs(child_docs: List[Document], parent_store):
    """根据子文档检索父文档"""
    parent_ids = {doc.metadata.get("parent_id") for doc in child_docs if doc and doc.metadata}
    if not parent_ids:
        return []
    return parent_store.mget(list(parent_ids))


def format_content(docs: List[Document]) -> str:
    """只格式化文档内容"""
    if not docs:
        return "未找到相关上下文信息。"
    return "\n\n".join(doc.page_content for doc in docs if doc)


def format_sources(docs: List[Document]) -> str:
    """只格式化来源信息"""
    if not docs:
        return ""
    sources = sorted({os.path.basename(doc.metadata.get("source", "未知来源")) for doc in docs if doc and doc.metadata})
    return ", ".join(sources)


def reciprocal_rank_fusion(list_of_doc_lists: list[list[Document]], k=60) -> List[Document]:
    """使用倒数排名融合（RRF）算法合并和重排文档"""
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
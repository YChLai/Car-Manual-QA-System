import pickle
import uuid

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import Chroma
from langchain_core.stores import InMemoryStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_paths=["./model_3.pdf","./model_y.pdf","./model_s.pdf"]
all_files=[]
for file_path in file_paths:
    loader = UnstructuredFileLoader(file_path)
    file=loader.load()
    all_files.extend(file)

print(f"所有文档加载完成，共 {len(all_files)} 个初始文档块。")

print("\n>>> 分割父文档...")

parent_splitter=RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100
)
parent_chunks=parent_splitter.split_documents(all_files)

#给每个父文档一个唯一的ID
doc_ids = [str(uuid.uuid4()) for _ in parent_chunks]

embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
child_splitter=SemanticChunker(
    embeddings=embedding,
    breakpoint_threshold_type="percentile"
)

print("\n>>> 分割子文档...")

#把每个父文档再次按语义切割成子文档，并给子文档绑定父文档id
child_chunks=[]
for i,enum in enumerate(parent_chunks):
    doc_id=doc_ids[i]
    _chunks=(child_splitter.split_documents([enum]))
    for child_chunk in _chunks:
        child_chunk.metadata["parent_id"]=doc_id
    #child_chunks.append(_chunks)，这样变成聚合成一个list，即{[子文档1A，子文档1B]，[子文档2A]}
    #而用extend可以{子文档1A，子文档1B，子文档2A},这样来得到一个包含所有子文档的、扁平的大列表
    child_chunks.extend(_chunks)

print(f"成功将文档分割成 {len(parent_chunks)} 个父文档块和 {len(child_chunks)} 个子文档块。")

parent_store=InMemoryStore()
#mset存储键值对
parent_store.mset(list(zip(doc_ids, parent_chunks)))
with open("./parent_store_data.pkl", "wb") as f:
    # pickle.dump需要两个参数，第一个是要保存的对象，第二个是文件句柄
    # 我们把父文档块和ID都保存在一个元组里
    pickle.dump((parent_chunks, doc_ids), f)


vectorstore_path = "./chroma_db"


child_vectorstore = Chroma.from_documents(
    documents=child_chunks,
    embedding=embedding,
    persist_directory=vectorstore_path
)
print(f"\n--- 索引保存成功！---")
print(f"父文档库已保存在: ./docstore")
print(f"子文档向量索引已保存在: {vectorstore_path}")







from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
import pandas as pd
import os
import unicodedata
import time
import logging
import tiktoken

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='hybrid_experiment.log', filemode='a')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_dataset(file_path):
    if file_path.endswith('.xlsx'):  # 파일 확장자에 따라 다른 방식으로 데이터셋 로드
        return pd.read_excel(file_path)
    else:
        return pd.read_csv(file_path)

def load_all_text_documents(folder_path):
    documents = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                documents.extend(TextLoader(file_path).load())
    return documents

def split_documents(documents, chunk_size, overlap_size):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=chunk_size,
        chunk_overlap=overlap_size
    )
    return text_splitter.split_documents(documents)

def create_vector_store(documents, embedding_model, chunk_size, overlap_size):
    persist_dir_name = f"new_vector_store_{embedding_model}_c{chunk_size}_o{overlap_size}"
    persist_dir_path = os.path.join("./vector_stores", persist_dir_name)
    if os.path.exists(persist_dir_path):
        logging.info(f"Vector store already exists at {persist_dir_path}. Skipping embedding.")
        vector_store = Chroma(persist_directory=persist_dir_path, embedding_function=OpenAIEmbeddings(model=embedding_model, openai_api_key=OPENAI_API_KEY))
    else:
        os.makedirs(persist_dir_path, exist_ok=True)
        start_time = time.time()
        vector_store = Chroma.from_documents(documents, OpenAIEmbeddings(model=embedding_model, openai_api_key=OPENAI_API_KEY), persist_directory=persist_dir_path)
        embedding_time = time.time() - start_time
        total_tokens = sum([len(tiktoken.encoding_for_model("text-embedding-3-small").encode(doc.page_content)) for doc in documents])
        logging.info(f"Embedding completed in {embedding_time:.2f} seconds for chunk size {chunk_size} and overlap size {overlap_size}. Total tokens embedded: {total_tokens}")
    return vector_store, persist_dir_path

def search_and_generate_answer(query, vector_db_300, vector_db_800):
    """
    Chunk 300 VectorDB에서 문서 이름을 검색한 후 Chunk 800 VectorDB에서 답변 생성
    """
    start_time = time.time()
    retrieved_docs_300 = vector_db_300.similarity_search(query, k=5)
    if not retrieved_docs_300:
        return "No relevant documents found in Chunk 300 VectorDB.", None, 0, 0
    
    # target_doc_name = retrieved_docs_300[0].metadata.get('source')
    # if not target_doc_name:
    #     return "Could not retrieve document name from Chunk 300 VectorDB.", None, 0, 0
    
        # similarity_search_with_score를 사용하여 검색 결과와 점수를 가져옴
    all_docs_with_scores = vector_db_800.similarity_search_with_score(query, k=50)  # 충분히 큰 k값 사용
    # 필터링하여 retrieved_docs_300에 해당하는 문서만 남김
    filtered_docs_with_scores = [
        (doc, score) for doc, score in all_docs_with_scores
        if any(retrieved_doc.metadata.get('source') in doc.metadata.get('source') for retrieved_doc in retrieved_docs_300)
    ]
    
    # 필터링된 결과에서 상위 5개 선택
    filtered_docs_with_scores = sorted(filtered_docs_with_scores, key=lambda x: x[1])[:5]
    filtered_docs = [doc for doc, _ in filtered_docs_with_scores]
    
    if not filtered_docs:
        return f"No relevant content found in Chunk 800 VectorDB for document {retrieved_docs_300.metadata.get('source')}.", None, 0, 0

    passages = "\n".join([f"Passage {i}: {doc.page_content}" for i, doc in enumerate(filtered_docs)])
    sources = [os.path.basename(unicodedata.normalize('NFC', doc.metadata['source'])).replace('.txt', '') for doc in filtered_docs]  # 확장자 제거

    prompt_template = f"""
# Question: {query}

# Relevant Passages:
{passages}

# You are tasked to answer questions based on specific provided passages. For each query:

Generate a one-sentence, concise response directly addressing the question.
Ensure the answer is precise, does not elaborate beyond what is asked, and is always backed by the passages provided.
"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["query", "passages"])
    chain = prompt | llm
    response = chain.invoke({"query": query, "passages": passages})

    tokens_used = (
        len(tiktoken.encoding_for_model("gpt-4o-mini").encode(query)) +
        len(tiktoken.encoding_for_model("gpt-4o-mini").encode(passages)) +
        len(tiktoken.encoding_for_model("gpt-4o-mini").encode(response.content))
    )
    response_time = time.time() - start_time

    return response.content, sources if sources else None, tokens_used, response_time

def save_results_with_rag(dataset, vector_db_300, vector_db_900):
    """
    데이터셋에 있는 질문에 대해 답변을 생성하고, 결과와 성능 데이터를 저장
    """
    dataset_copy = dataset.copy()
    rag_results = []
    sources_list = []
    total_tokens_used = 0
    total_time_taken = 0

    for query in dataset["question"]:
        predicted_answer, sources, tokens_used, response_time = search_and_generate_answer(query, vector_db_300, vector_db_900)
        rag_results.append(predicted_answer)
        sources_list.append(", ".join(sources) if sources else "N/A")
        total_tokens_used += tokens_used
        total_time_taken += response_time

    logging.info(f"Total time to generate answers: {total_time_taken:.2f} seconds. Total tokens used: {total_tokens_used}.")
    dataset_copy["result"] = rag_results
    dataset_copy["source"] = sources_list

    output_file = f"./results/hybrid_dataset.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    dataset_copy.to_csv(output_file, index=False, encoding='utf-8-sig')
    logging.info(f"Saved results to {output_file}")

def run_experiment(folder_path, dataset_path, embedding_model="text-embedding-3-small"):
    logging.info(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)
    logging.info(f"Loading all text documents from folder {folder_path}")
    documents = load_all_text_documents(folder_path)

    # Chunk 300 VectorDB 생성
    logging.info(f"Creating Chunk 300 VectorDB")
    split_docs_300 = split_documents(documents, chunk_size=300, overlap_size=50)
    vector_db_300, _ = create_vector_store(split_docs_300, embedding_model, chunk_size=300, overlap_size=50)

    # Chunk 800 VectorDB 생성
    logging.info(f"Creating Chunk 900 VectorDB")
    split_docs_900 = split_documents(documents, chunk_size=900, overlap_size=50)
    vector_db_900, _ = create_vector_store(split_docs_900, embedding_model, chunk_size=900, overlap_size=50)

    # 질문 처리 및 답변 생성
    save_results_with_rag(dataset, vector_db_300, vector_db_900)
    logging.info("Experiment completed.")


# 실험 파라미터
folder_path = "./data_txt"
dataset_path = "./data_txt/dataset.xlsx"

run_experiment(folder_path, dataset_path)

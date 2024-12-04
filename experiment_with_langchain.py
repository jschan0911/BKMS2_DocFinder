from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from sklearn.metrics import f1_score
import pandas as pd
import os
import unicodedata
import time
import logging
import tiktoken

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='experiment.log', filemode='a')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_dataset(file_path):
    if file_path.endswith('.xlsx'): # 파일 확장자에 따라 다른 방식으로 데이터셋 로드
        return pd.read_excel(file_path)
    else:
        return pd.read_csv(file_path)

# data 폴더 아래에 있는 모든 텍스트 파일을 로드하는 함수
def load_all_text_documents(folder_path):
    documents = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                documents.extend(TextLoader(file_path).load())
    return documents

# 청크 크기와 중첩 크기에 따른 문서 분할 함수
def split_documents(documents, chunk_size, overlap_size):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size)
    return text_splitter.split_documents(documents)

# 벡터 저장소를 생성하고 실험 후에도 저장
def create_vector_store(documents, embedding_model, chunk_size, overlap_size):
    persist_dir_name = f"vector_store_{embedding_model}_c{chunk_size}_o{overlap_size}"
    persist_dir_path = os.path.join("./vector_stores", persist_dir_name)
    if os.path.exists(persist_dir_path):
        logging.info(f"Vector store already exists at {persist_dir_path}. Skipping embedding.")
        vector_store = Chroma(persist_directory=persist_dir_path, embedding_function=OpenAIEmbeddings(model=embedding_model, openai_api_key=OPENAI_API_KEY))
    else:
        os.makedirs(persist_dir_path, exist_ok=True)
        start_time = time.time()
        vector_store = Chroma.from_documents(documents, OpenAIEmbeddings(model=embedding_model, openai_api_key=OPENAI_API_KEY), persist_directory=persist_dir_path)
        embedding_time = time.time() - start_time

        # 특정 모델을 사용했을 때의 총 토큰 수 계산
        total_tokens = sum([len(tiktoken.encoding_for_model("text-embedding-3-small").encode(doc.page_content)) for doc in documents])
        logging.info(f"Embedding completed in {embedding_time:.2f} seconds for chunk size {chunk_size} and overlap size {overlap_size}. Total tokens embedded: {total_tokens}")
    return vector_store, persist_dir_path

# 쿼리에 대해 벡터 저장소에서 유사한 문서를 검색하고 답변 생성
def generate_answer(query, vector_store):
    retrieved_docs = vector_store.similarity_search(query, k=5)
    passages = "\n".join([f"Passage {i}: {doc.page_content}" for i, doc in enumerate(retrieved_docs)])  # 출처 제거
    sources = [os.path.basename(unicodedata.normalize('NFC', doc.metadata['source'])).replace('.txt', '') for doc in retrieved_docs]  # 확장자 제거

    prompt_template = f"""
# Question: {query}

# Relevant Passages:
{passages}

# You are tasked to answer questions based on specific provided passages. For each query:

Generate a one-sentence, concise response directly addressing the question.
Ensure the answer is precise, does not elaborate beyond what is asked, and is always backed by the passages provided.
"""
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
        prompt = PromptTemplate(template=prompt_template, input_variables=["query", "passages"])
        chain = prompt | llm
        response = chain.invoke({"query": query, "passages": passages})

        # 특정 모델을 사용했을 때의 총 토큰 수 계산
        tokens_used = len(tiktoken.encoding_for_model("gpt-4o-mini").encode(query)) + len(tiktoken.encoding_for_model("gpt-4o-mini").encode(passages)) + len(tiktoken.encoding_for_model("gpt-4o-mini").encode(response.content))
        return response.content, sources if sources else None, tokens_used
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        return None, None

def save_results_with_rag(dataset, vector_store, chunk_size, overlap_size, vectordb_size):
    dataset_copy = dataset.copy()
    rag_results = []
    sources_list = []
    total_tokens_used = 0

    start_time = time.time()
    for query in dataset["question"]:
        predicted_answer, sources, tokens_used = generate_answer(query, vector_store)
        rag_results.append(predicted_answer)
        sources_list.append(", ".join(sources) if sources else "N/A")
        total_tokens_used += tokens_used
    total_time = time.time() - start_time
    logging.info(f"Total time to generate all answers: {total_time:.2f} seconds for Chunk Size: {chunk_size}, Overlap Size: {overlap_size}, Total tokens used for generating response: {total_tokens_used}")


    dataset_copy["result"] = rag_results
    dataset_copy["source"] = sources_list  # 출처 추가
    
    output_file = f"./results/c{chunk_size}_o{overlap_size}_{int(vectordb_size / (1024 * 1024))}MB_dataset.csv"
    dataset_copy.to_csv(output_file, index=False, encoding='utf-8-sig')  # CSV로 저장, 한글 깨짐 방지
    logging.info(f"Saved results to {output_file}")

# 디렉토리의 전체 크기를 계산하는 함수
def get_directory_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def run_experiment(folder_path, dataset_path, chunk_sizes, overlap_sizes, embedding_model="text-embedding-3-small"):
    # 데이터셋 로드
    logging.info(f"Loading dataset from {dataset_path}")
    dataset = load_dataset(dataset_path)

    # 모든 텍스트 문서 로드
    logging.info(f"Loading all text documents from folder {folder_path}")
    documents = load_all_text_documents(folder_path)

    # 청크 크기와 중첩 크기의 조합을 반복하여 실험
    for chunk_size in chunk_sizes:
        for overlap_size in overlap_sizes:
            try:
                logging.info(f"\n--- Running experiment with Chunk Size: {chunk_size}, Overlap Size: {overlap_size} ---")
                # 문서 분할
                split_docs = split_documents(documents, chunk_size, overlap_size)
                # 벡터 저장소 생성 및 영구 저장
                vector_store, persist_dir_path = create_vector_store(split_docs, embedding_model, chunk_size, overlap_size)
                # 청크와 중첩 크기에 따른 vectordb 저장소 크기 출력
                vectordb_size = get_directory_size(persist_dir_path)
                logging.info(f"VectorDB Storage Size: {vectordb_size / (1024 * 1024):.2f} MB for Chunk Size: {chunk_size}, Overlap Size: {overlap_size}")
                # RAG 결과 저장
                save_results_with_rag(dataset, vector_store, chunk_size, overlap_size, vectordb_size)
            except ValueError as ve:
                logging.error(f"ValueError occurred for Chunk Size: {chunk_size}, Overlap Size: {overlap_size}: {ve}")
                continue  # 다음 실험으로 넘어감
            except Exception as e:
                logging.error(f"Unexpected error occurred for Chunk Size: {chunk_size}, Overlap Size: {overlap_size}: {e}")
                continue  # 다음 실험으로 넘어감


# 실험 파라미터 설정
# folder_path = "./data"    # 테스트 파일 경로
# dataset_path = "./data/dataset.csv"  # 테스트 데이터셋 경로

folder_path = "./data_txt"
dataset_path = "./data_txt/dataset.xlsx"

# 실험할 청크 크기
chunk_sizes = list(range(300, 2100, 100))  # 100부터 1000까지 100 단위로 채움

# 실험할 중첩 크기
overlap_sizes = [50]

run_experiment(folder_path, dataset_path, chunk_sizes, overlap_sizes)

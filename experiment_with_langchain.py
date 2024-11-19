from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from sklearn.metrics import f1_score
import pandas as pd
import os
import shutil
import tempfile

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

# 임시 폴더를 사용해 임베딩을 저장하고 실험 후 삭제
def create_vector_store(documents, embedding_model):
    temp_dir = tempfile.mkdtemp()  # 임시 디렉터리 생성
    vector_store = Chroma.from_documents(documents, OpenAIEmbeddings(model=embedding_model, openai_api_key=OPENAI_API_KEY), persist_directory=temp_dir)
    return vector_store, temp_dir

# 쿼리에 대해 벡터 저장소에서 유사한 문서를 검색하고 답변 생성
def generate_answer(query, vector_store):
    retrieved_docs = vector_store.similarity_search(query, k=5)
    passages = "\n".join([f"Passage {i} (data_source: {doc.metadata['source']}):\n{doc.page_content}\n" for i, doc in enumerate(retrieved_docs)])
    prompt_template = f"""
# Question: {query}

# Relevant Passages:
{passages}

# Based on the passages above, generate an answer to the question. Explicitly mention the 'data_source'.
ex) (출처: gsds_notification.pdf)
"""
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
        prompt = PromptTemplate(template=prompt_template, input_variables=["query", "passages"])
        chain = prompt | llm
        response = chain.invoke({"query": query, "passages": passages})
        return response.content
    except Exception as e:
        print(e)
        return None

# # 정확도 계산 함수
# def calculate_accuracy(dataset, vector_store):
#     true_answers = dataset['answer']
#     queries = dataset['question']
#     predicted_answers = []

#     for query in queries:
#         predicted_answer = generate_answer(query, vector_store)
#         predicted_answers.append(predicted_answer)

#     # 간단한 문자열 일치로 정확도 계산
#     correct = 0
#     for true, pred in zip(true_answers, predicted_answers):
#         if true.strip() in pred:
#             correct += 1

#     accuracy = correct / len(true_answers)
#     return accuracy

# 예상 답안과 데이터셋의 정답 비교 출력 함수
def compare_answers(dataset, vector_store):
    true_answers = dataset['answer']
    queries = dataset['question']

    for query, true_answer in zip(queries, true_answers):
        predicted_answer = generate_answer(query, vector_store)
        print(f"\nQuestion: {query}")
        print(f"Expected Answer: {true_answer}")
        print(f"Predicted Answer: {predicted_answer}")

def save_results_with_rag(dataset, vector_store, chunk_size, overlap_size, vectordb_size):
    dataset_copy = dataset.copy()
    rag_results = []

    for query in dataset["question"]:
        predicted_answer = generate_answer(query, vector_store)
        rag_results.append(predicted_answer)

    dataset_copy["result"] = rag_results

    # output_file = f"./data_txt/c{chunk_size}_o{overlap_size}_{int(vectordb_size / (1024 * 1024))}MB_dataset.csv"
    # dataset_copy.to_csv(output_file, index=False) # csv로 저장
    
    output_file = f"./data_txt/c{chunk_size}_o{overlap_size}_{int(vectordb_size / (1024 * 1024))}MB_dataset.xlsx"
    dataset_copy.to_excel(output_file, index=False, engine='openpyxl')  # xlsx로 저장
    print(f"Saved results to {output_file}")

# 디렉토리의 전체 크기를 계산하는 함수
def get_directory_size(directory):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

# 다양한 청크 크기와 중첩 크기 조합에 따른 실험 수행
def run_experiment(folder_path, dataset_path, chunk_sizes, overlap_sizes, embedding_model="text-embedding-3-large"):
    # 데이터셋 로드
    dataset = load_dataset(dataset_path)

    # 모든 텍스트 문서 로드
    documents = load_all_text_documents(folder_path)

    # 청크 크기와 중첩 크기의 조합을 반복하여 실험
    for chunk_size in chunk_sizes:
        for overlap_size in overlap_sizes:
            print(f"\n--- Chunk Size: {chunk_size}, Overlap Size: {overlap_size} ---")
            # 문서 분할
            split_docs = split_documents(documents, chunk_size, overlap_size)
            # 벡터 저장소 생성
            vector_store, temp_dir = create_vector_store(split_docs, embedding_model)
            # 청크와 중첩 크기에 따른 vectordb 저장소 크기 출력
            vectordb_size = get_directory_size(temp_dir)
            # RAG 결과 저장
            save_results_with_rag(dataset, vector_store, chunk_size, overlap_size, vectordb_size)
            # 예상 답안과 비교 출력
            # compare_answers(dataset, vector_store)
            print(f"VectorDB Storage Size: {vectordb_size / (1024 * 1024):.2f} MB")
            # 임시 폴더 삭제
            shutil.rmtree(temp_dir)

# 실험 파라미터 설정

# folder_path = "./data"    # 테스트 파일 경로
# dataset_path = "./data_txt/dataset.csv"  # 테스트 데이터셋 경로

folder_path = "./data_txt"  # 실험 파일 경로
dataset_path = "./data_txt/dataset.xlsx"  # 실험 데이터셋 경로
chunk_sizes = [500]  # 실험할 청크 크기
overlap_sizes = [50]  # 실험할 중첩 크기

run_experiment(folder_path, dataset_path, chunk_sizes, overlap_sizes)

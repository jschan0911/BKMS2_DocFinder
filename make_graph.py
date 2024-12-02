import pandas as pd
import matplotlib.pyplot as plt

# Column 명
CHUNK_SIZE = 'Chunk Size'
EMBEDDING_TIME = 'Embedding Time(sec)'
EMBEDDING_TOKEN = 'Embedding Token'
VECTORDB_SIZE = 'VectorDB Size(MB)'
LLM_RESPONSE_TIME = 'LLM Response Time(sec)'
LLM_RESPONSE_TOKEN = 'LLM Response Token'
BERT_F1 = 'BERT F1'
F1 = 'F1'
CONTAIN_COUNT = 'Contain Count'


df_original = pd.read_csv('./data_result_csv/chunk_size_analysis.csv')
df_splitter = pd.read_csv('./data_result_csv/chunk_size_with_linebreaks.csv')

# 1. 청크 크기에 따른 Vector DB 크기 비교
plt.figure(figsize=(10, 6))
plt.plot(df_original[CHUNK_SIZE], df_original[VECTORDB_SIZE], label=VECTORDB_SIZE)
plt.title("Vector DB Size by Chunk Size")
plt.xlabel("Chunk Size")
plt.ylabel("Vector DB Size (MB)")
plt.legend()
plt.grid(True)
plt.savefig('./graphs/vector_db_size_by_chunk_size.png')

# # 2. 청크 크기에 따른 임베딩 시간 + 응답 생성 시간 비교
# #   - 임베딩 시간의 차이는 미미했음
# #   - 그러나 LLM을 활용하여 응답을 생성하는 시간은 크게 차이가 남
# plt.figure(figsize=(10, 6))
# plt.plot(df_original[CHUNK_SIZE], df_original[EMBEDDING_TIME] + df_original[LLM_RESPONSE_TIME], label="Whole Time")
# plt.title("Embedding Time and LLM Response Time by Chunk Size")
# plt.xlabel("Chunk Size")
# plt.ylabel("Time (sec)")
# plt.legend()
# plt.grid(True)
# plt.savefig('./graphs/embedding_time_and_llm_response_time_by_chunk_size.png')

# # 3. 청크 크기에 따른 토큰 사용량 비교 
# #  - 임베딩 토큰 수는 청크가 커질 수록 지수함수적으로 감소
# #  - LLM 응답 토큰 수는 청크가 커질 수록 선형적으로 증가
# plt.figure(figsize=(10, 6))
# # plt.plot(df_original[CHUNK_SIZE], df_original[EMBEDDING_TOKEN], label=EMBEDDING_TOKEN)
# plt.plot(df_original[CHUNK_SIZE], df_original[LLM_RESPONSE_TOKEN], label=LLM_RESPONSE_TOKEN)
# plt.title("Token Usage by Chunk Size")
# plt.xlabel("Chunk Size")
# plt.ylabel("Token Usage")
# plt.legend()
# plt.grid(True)
# # plt.show()
# plt.savefig('./graphs/llm_token_usage_by_chunk_size.png')

# # 4. 청크 크기에 따른 총 비용 비교
# #  - 총 비용에는 LLM 응답에 사용된 비용이 많은 영향을 미침 (5배)
# #  - 따라서 청크 크기가 커질 수록 
# #    - 임베딩 토큰 수가 지수함수적으로 줄었음에도 불구하고, 
# #    - 응답 생성 비용의 영향력이 커서 총 비용이 증가함

# # Define the cost per token for embedding and LLM response
# EMBEDDING_COST_PER_TOKEN = 0.00002 / 1000  # $0.00002 per 1,000 tokens
# LLM_COST_PER_TOKEN = 0.0001 / 1000  # $0.0001 per 1,000 tokens

# # Add cost columns to the dataframe
# df_original['Embedding Cost'] = df_original[EMBEDDING_TOKEN] * EMBEDDING_COST_PER_TOKEN
# df_original['LLM Response Cost'] = df_original[LLM_RESPONSE_TOKEN] * LLM_COST_PER_TOKEN
# df_original['Total Cost'] = df_original['Embedding Cost'] + df_original['LLM Response Cost']

# # Plot total cost by chunk size
# plt.figure(figsize=(10, 6))
# # plt.plot(df_original[CHUNK_SIZE], df_original['Total Cost'], label='Total Cost')
# plt.plot(df_original[CHUNK_SIZE], df_original['LLM Response Cost'], label='LLM Response Cost')
# plt.title("LLM Response Cost by Chunk Size")
# plt.xlabel("Chunk Size")
# plt.ylabel("Cost (USD)")
# plt.legend()
# plt.grid(True)


# # 5. 청크 수에 따른 BERT F1, F1, Contain Count 비교
# #    - 청크 수가 증가할 수록 BERT F1, F1은 증가
# #    - Contain Count는 청크 수가 증가할 수록 감소

# plt.figure(figsize=(10, 6))
# plt.plot(df_original[CHUNK_SIZE], df_original[BERT_F1], label=BERT_F1)
# plt.plot(df_original[CHUNK_SIZE], df_original[F1], label=F1)
# # plt.plot(df_original[CHUNK_SIZE], df_original[CONTAIN_COUNT], label=CONTAIN_COUNT)
# plt.title("Accuracy by Chunk Size")
# plt.xlabel("Chunk Size")
# plt.ylabel("Value")
# plt.legend()
# plt.grid(True)
# # plt.show()
# plt.savefig('./graphs/accuracy_contain_count.png')

# # 6. 줄바꿈을 포함한 청크 수에 따른 BERT F1, F1, Contain Count 비교
# plt.figure(figsize=(10, 6))
# # plt.plot(df_splitter[CHUNK_SIZE], df_splitter[BERT_F1], label=BERT_F1)
# # plt.plot(df_splitter[CHUNK_SIZE], df_splitter[F1], label=F1)
# plt.plot(df_splitter[CHUNK_SIZE], df_splitter[CONTAIN_COUNT], label=CONTAIN_COUNT)
# plt.title("Accuracy by Chunk Size (Line Breaks)")
# plt.xlabel("Chunk Size")
# plt.ylabel("Value")
# plt.legend()
# plt.grid(True)
# # plt.show()
# plt.savefig('./graphs/accuracy_linebreaks_contain_count.png')

# # 7. 청크 방식에 따른 BERT F1, F1, Contain Count 비교
# plt.figure(figsize=(10, 6))
# # plt.plot(df_original[CHUNK_SIZE], df_original[BERT_F1], label='BERT F1 (Original)')
# # plt.plot(df_splitter[CHUNK_SIZE], df_splitter[BERT_F1], label='BERT F1 (Line Breaks)')

# # plt.plot(df_original[CHUNK_SIZE], df_original[F1], label='F1 (Original)')
# # plt.plot(df_splitter[CHUNK_SIZE], df_splitter[F1], label='F1 (Line Breaks)')

# plt.plot(df_original[CHUNK_SIZE], df_original[CONTAIN_COUNT], label='Contain Count (Original)')
# plt.plot(df_splitter[CHUNK_SIZE], df_splitter[CONTAIN_COUNT], label='Contain Count (Line Breaks)')

# plt.title("Accuracy by Chunk Size")
# plt.xlabel("Chunk Size")
# plt.ylabel("Value")
# plt.legend()
# plt.grid(True)
# # plt.show()
# plt.savefig('./graphs/accuracy_comparison_contain_count.png')


# # 8. 줄바꿈 포함 청크 방식에 따른 시간 비교
# plt.figure(figsize=(10, 6))
# # plt.plot(df_splitter[CHUNK_SIZE], df_splitter[EMBEDDING_TIME], label='Embedding Time (Line Breaks)')
# plt.plot(df_splitter[CHUNK_SIZE], df_splitter[LLM_RESPONSE_TIME], label='LLM Response Time (Line Breaks)')
# plt.title("Time by Chunk Size (Line Breaks)")
# plt.xlabel("Chunk Size")
# plt.ylabel("Time (sec)")
# plt.legend()
# plt.grid(True)
# # plt.show()
# plt.savefig('./graphs/time_linebreaks_response.png')
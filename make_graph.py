import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Column 명
CHUNK_SIZE = 'Chunk Size'
EMBEDDING_TIME = 'Embedding Time(sec)'
EMBEDDING_TOKEN = 'Embedding Token'
VECTORDB_SIZE = 'VectorDB Size(MB)'
LLM_RESPONSE_TIME = 'LLM Response Time(sec)'
LLM_RESPONSE_TOKEN = 'LLM Response Token'
BERT_F1 = 'BERT F1'
F1 = 'F1'
COSINE_SIMILARITY = 'Cosine'
CONTAIN_COUNT = 'Contain Count'


df_original = pd.read_csv('./data_result_csv/chunk_size_analysis.csv')
df_splitter = pd.read_csv('./data_result_csv/chunk_size_with_linebreaks.csv')
df_hybrid = pd.read_csv('./data_result_csv/hybrid.csv')

# # 1. 청크 크기에 따른 Vector DB 크기 비교
# plt.figure(figsize=(10, 6))
# plt.plot(df_original[CHUNK_SIZE], df_original[VECTORDB_SIZE], label=VECTORDB_SIZE)
# plt.title("Vector DB Size by Chunk Size")
# plt.xlabel("Chunk Size")
# plt.ylabel("Vector DB Size (MB)")
# plt.legend()
# plt.grid(True)
# plt.savefig('./graphs/vector_db_size_by_chunk_size.png')

# # 2. 청크 크기에 따른 임베딩 시간 + 응답 생성 시간 비교
# #   - 임베딩 시간의 차이는 미미했음
# #   - 그러나 LLM을 활용하여 응답을 생성하는 시간은 크게 차이가 남
# plt.figure(figsize=(10, 6))
# # plt.plot(df_original[CHUNK_SIZE], df_original[LLM_RESPONSE_TIME], label="Whole Time")
# plt.plot(df_splitter[CHUNK_SIZE], df_splitter[LLM_RESPONSE_TIME], label="Whole Time (Line Breaks)")
# plt.title("Embedding Time and LLM Response Time by Chunk Size")
# plt.xlabel("Chunk Size")
# plt.ylabel("Time (sec)")
# plt.legend()
# plt.grid(True)
# # plt.savefig('./graphs/embedding_time_and_llm_response_time_by_chunk_size.png')
# plt.show()

# # 3. 청크 크기에 따른 토큰 사용량 비교 
# #  - 임베딩 토큰 수는 청크가 커질 수록 지수함수적으로 감소
# #  - LLM 응답 토큰 수는 청크가 커질 수록 선형적으로 증가
# scaler = MinMaxScaler()
# normalized_df = df_splitter.copy()
# normalized_df[EMBEDDING_TOKEN] = scaler.fit_transform(df_splitter[[EMBEDDING_TOKEN]])
# normalized_df[LLM_RESPONSE_TOKEN] = scaler.fit_transform(df_splitter[[LLM_RESPONSE_TOKEN]])

# plt.figure(figsize=(10, 6))
# plt.plot(normalized_df[CHUNK_SIZE], normalized_df[EMBEDDING_TOKEN], label=EMBEDDING_TOKEN+' (Normalized)')
# plt.plot(normalized_df[CHUNK_SIZE], normalized_df[LLM_RESPONSE_TOKEN], label=LLM_RESPONSE_TOKEN+' (Normalized)')
# plt.title("Token Usage by Chunk Size")
# plt.xlabel("Chunk Size")
# plt.ylabel("Token Usage")
# plt.legend()
# plt.grid(True)
# # plt.show()
# plt.savefig('./graphs/token_usages_by_chunk_size(normalized).png')

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
# # plt.plot(df_original[CHUNK_SIZE], df_original['LLM Response Cost'], label='LLM Response Cost')
# plt.plot(df_original[CHUNK_SIZE], df_original['Total Cost'], label='Total Cost')
# plt.title("Total Cost by Chunk Size")
# plt.xlabel("Chunk Size")
# plt.ylabel("Cost (USD)")
# plt.legend()
# plt.grid(True)
# plt.show()


# # # 5. 청크 수에 따른 BERT F1, F1, Contain Count 비교
# #    - 청크 수가 증가할 수록 BERT F1, F1은 증가
# #    - Contain Count는 청크 수가 증가할 수록 감소

# # 데이터 정규화
# scaler = MinMaxScaler()
# normalized_df = df_original.copy()
# normalized_df[BERT_F1] = scaler.fit_transform(df_original[[BERT_F1]])
# normalized_df[F1] = scaler.fit_transform(df_original[[F1]])
# normalized_df[COSINE_SIMILARITY] = scaler.fit_transform(df_original[[COSINE_SIMILARITY]])
# normalized_df[CONTAIN_COUNT] = scaler.fit_transform(df_original[[CONTAIN_COUNT]])

# # Plot the normalized data
# plt.figure(figsize=(10, 6))
# plt.plot(normalized_df[CHUNK_SIZE], normalized_df[BERT_F1], label=f"Normalized {BERT_F1}")
# plt.plot(normalized_df[CHUNK_SIZE], normalized_df[F1], label=f"Normalized {F1}")
# plt.plot(normalized_df[CHUNK_SIZE], normalized_df[COSINE_SIMILARITY], label=f"Normalized {COSINE_SIMILARITY}")
# plt.plot(normalized_df[CHUNK_SIZE], normalized_df[CONTAIN_COUNT], label=f"Normalized {CONTAIN_COUNT}")
# plt.title("Accuracy by Chunk Size")
# plt.xlabel("Chunk Size")
# plt.ylabel("Value")
# plt.legend()
# plt.grid(True)
# # plt.show()
# plt.savefig('./graphs/accuracy_comparison_normalized.png')

# # 6. 줄바꿈을 포함한 청크 수에 따른 BERT F1, F1, Contain Count 비교
# scaler = MinMaxScaler()
# normalized_df = df_splitter.copy()
# normalized_df[BERT_F1] = scaler.fit_transform(df_splitter[[BERT_F1]])
# normalized_df[F1] = scaler.fit_transform(df_splitter[[F1]])
# normalized_df[CONTAIN_COUNT] = scaler.fit_transform(df_splitter[[CONTAIN_COUNT]])

# plt.figure(figsize=(10, 6))
# plt.plot(normalized_df[CHUNK_SIZE], normalized_df[BERT_F1], label=f"Normalized {BERT_F1}")
# plt.plot(normalized_df[CHUNK_SIZE], normalized_df[F1], label=f"Normalized {F1}")
# plt.plot(normalized_df[CHUNK_SIZE], normalized_df[CONTAIN_COUNT], label=f"Normalized {CONTAIN_COUNT}")
# plt.title("Accuracy by Chunk Size (Line Breaks)")
# plt.xlabel("Chunk Size")
# plt.ylabel("Value")
# plt.legend()
# plt.grid(True)
# # plt.show()
# plt.savefig('./graphs/accuracy_linebreaks_contain_count.png')

# # 7. 청크 방식에 따른 BERT F1, F1, Contain Count 비교
# plt.figure(figsize=(10, 6))
# plt.plot(df_original[CHUNK_SIZE], df_original[BERT_F1], label='BERT F1 (Original)')
# plt.plot(df_splitter[CHUNK_SIZE], df_splitter[BERT_F1], label='BERT F1 (Line Breaks)')

# # plt.plot(df_original[CHUNK_SIZE], df_original[F1], label='F1 (Original)')
# # plt.plot(df_splitter[CHUNK_SIZE], df_splitter[F1], label='F1 (Line Breaks)')

# # plt.plot(df_original[CHUNK_SIZE], df_original[COSINE_SIMILARITY], label='Cosine Similarity (Original)')
# # plt.plot(df_splitter[CHUNK_SIZE], df_splitter[COSINE_SIMILARITY], label='Cosine Similarity (Line Breaks)')

# # plt.plot(df_original[CHUNK_SIZE], df_original[CONTAIN_COUNT], label='Contain Count (Original)')
# # plt.plot(df_splitter[CHUNK_SIZE], df_splitter[CONTAIN_COUNT], label='Contain Count (Line Breaks)')

# plt.title("Accuracy by Chunk Size")
# plt.xlabel("Chunk Size")
# plt.ylabel("Value")
# plt.legend()
# plt.grid(True)
# # plt.show()
# plt.savefig('./graphs/accuracy_comparison_bert.png')


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


# # 9. 청크 방식에 따른 Vector DB 크기 차이 비교
# # 차이 계산
# vector_db_diff = df_splitter[VECTORDB_SIZE] - df_original[VECTORDB_SIZE]

# # 차이 값 플롯
# plt.figure(figsize=(10, 6))
# plt.plot(df_original[CHUNK_SIZE], vector_db_diff, label='linebreaks - original')
# plt.axhline(0, color='red', linewidth=2, linestyle='--')
# plt.title("Difference in Vector DB Size by Chunk Size")
# plt.xlabel("Chunk Size")
# plt.ylabel("Difference in Vector DB Size (MB)")
# plt.legend()
# plt.grid(True)
# # plt.show()
# plt.savefig('./graphs/vector_db_size_difference_by_chunk_size.png')

# # 10. 청크 방식에 따른 총 비용 차이 비교
# # Define the cost per token for embedding and LLM response
# EMBEDDING_COST_PER_TOKEN = 0.00002 / 1000  # $0.00002 per 1,000 tokens
# LLM_COST_PER_TOKEN = 0.0001 / 1000  # $0.0001 per 1,000 tokens

# # Add cost columns to the dataframe
# df_original['Embedding Cost'] = df_original[EMBEDDING_TOKEN] * EMBEDDING_COST_PER_TOKEN
# df_original['LLM Response Cost'] = df_original[LLM_RESPONSE_TOKEN] * LLM_COST_PER_TOKEN
# df_original['Total Cost'] = df_original['Embedding Cost'] + df_original['LLM Response Cost']

# df_splitter['Embedding Cost'] = df_splitter[EMBEDDING_TOKEN] * EMBEDDING_COST_PER_TOKEN
# df_splitter['LLM Response Cost'] = df_splitter[LLM_RESPONSE_TOKEN] * LLM_COST_PER_TOKEN
# df_splitter['Total Cost'] = df_splitter['Embedding Cost'] + df_splitter['LLM Response Cost']

# # Plot total cost by chunk size
# plt.figure(figsize=(10, 6))
# plt.plot(df_original[CHUNK_SIZE], df_splitter['Total Cost'] - df_original['Total Cost'], label='linebreaks - original')
# plt.axhline(0, color='red', linewidth=2, linestyle='--')
# plt.title("Difference in Total Cost by Chunk Size")
# plt.xlabel("Chunk Size")
# plt.ylabel("Cost (USD)")
# plt.legend()
# plt.grid(True)
# # plt.show()
# plt.savefig('./graphs/total_cost_difference_by_chunk_size.png')

# # 11. 줄바꿈 단위로 청킹할 때 전체 정확도 비교(정규화)
# scaler = MinMaxScaler()
# normalized_df = df_splitter.copy()
# normalized_df[BERT_F1] = scaler.fit_transform(df_splitter[[BERT_F1]])
# normalized_df[F1] = scaler.fit_transform(df_splitter[[F1]])
# normalized_df[COSINE_SIMILARITY] = scaler.fit_transform(df_splitter[[COSINE_SIMILARITY]])
# normalized_df[CONTAIN_COUNT] = scaler.fit_transform(df_splitter[[CONTAIN_COUNT]])

# plt.figure(figsize=(10, 6))
# plt.plot(normalized_df[CHUNK_SIZE], normalized_df[BERT_F1], label=f"Normalized {BERT_F1}")
# plt.plot(normalized_df[CHUNK_SIZE], normalized_df[F1], label=f"Normalized {F1}")
# plt.plot(normalized_df[CHUNK_SIZE], normalized_df[COSINE_SIMILARITY], label=f"Normalized {COSINE_SIMILARITY}")
# plt.plot(normalized_df[CHUNK_SIZE], normalized_df[CONTAIN_COUNT], label=f"Normalized {CONTAIN_COUNT}")
# plt.title("Accuracy by Chunk Size (Line Breaks)")
# plt.xlabel("Chunk Size")
# plt.ylabel("Value")
# plt.legend()
# plt.grid(True)
# # plt.show()
# plt.savefig('./graphs/accuracy_linebreaks_normalized.png')


# 12. Hybrid 시스템과의 비교: Embedding Time(sec),Embedding Token,VectorDB Size(MB),LLM Response Time(sec),LLM Response Token를 비교
scaler = MinMaxScaler()
normalized_df = df_hybrid.copy()
normalized_df[EMBEDDING_TIME] = scaler.fit_transform(df_hybrid[[EMBEDDING_TIME]])
normalized_df[EMBEDDING_TOKEN] = scaler.fit_transform(df_hybrid[[EMBEDDING_TOKEN]])
normalized_df[VECTORDB_SIZE] = scaler.fit_transform(df_hybrid[[VECTORDB_SIZE]])
normalized_df[LLM_RESPONSE_TIME] = scaler.fit_transform(df_hybrid[[LLM_RESPONSE_TIME]])
normalized_df[LLM_RESPONSE_TOKEN] = scaler.fit_transform(df_hybrid[[LLM_RESPONSE_TOKEN]])

plt.figure(figsize=(10, 6))
# plt.plot(normalized_df[CHUNK_SIZE], normalized_df[EMBEDDING_TIME], label=EMBEDDING_TIME)
# plt.plot(normalized_df[CHUNK_SIZE], normalized_df[EMBEDDING_TOKEN], label=EMBEDDING_TOKEN)
# plt.plot(normalized_df[CHUNK_SIZE], normalized_df[VECTORDB_SIZE], label=VECTORDB_SIZE)
# plt.plot(normalized_df[CHUNK_SIZE], normalized_df[LLM_RESPONSE_TIME], label=LLM_RESPONSE_TIME)
# plt.plot(normalized_df[CHUNK_SIZE], normalized_df[LLM_RESPONSE_TOKEN], label=LLM_RESPONSE_TOKEN)

# plt.plot(df_hybrid[CHUNK_SIZE], df_hybrid[EMBEDDING_TIME], label=EMBEDDING_TIME)
# plt.plot(df_hybrid[CHUNK_SIZE], df_hybrid[EMBEDDING_TOKEN], label=EMBEDDING_TOKEN)
# plt.plot(df_hybrid[CHUNK_SIZE], df_hybrid[VECTORDB_SIZE], label=VECTORDB_SIZE)
# plt.plot(df_hybrid[CHUNK_SIZE], df_hybrid[LLM_RESPONSE_TIME], label=LLM_RESPONSE_TIME)
plt.plot(df_hybrid[CHUNK_SIZE], df_hybrid[LLM_RESPONSE_TOKEN], label=LLM_RESPONSE_TOKEN)
plt.ylim(0, max(df_hybrid[LLM_RESPONSE_TOKEN]) * 1.1)
plt.xlabel("Chunk Size")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
# plt.savefig('./graphs/hybrid_system_comparison_log.png')

# # 13. Hybrid 시스템과의 비교: BERT F1, F1, Contain Count 비교
# scaler = MinMaxScaler()
# normalized_df = df_hybrid.copy()
# normalized_df[BERT_F1] = scaler.fit_transform(df_hybrid[[BERT_F1]])
# normalized_df[F1] = scaler.fit_transform(df_hybrid[[F1]])
# normalized_df[COSINE_SIMILARITY] = scaler.fit_transform(df_hybrid[[COSINE_SIMILARITY]])
# normalized_df[CONTAIN_COUNT] = scaler.fit_transform(df_hybrid[[CONTAIN_COUNT]])

# plt.figure(figsize=(10, 6))
# plt.plot(normalized_df[CHUNK_SIZE], normalized_df[BERT_F1], label=BERT_F1)
# plt.plot(normalized_df[CHUNK_SIZE], normalized_df[F1], label=F1)
# plt.plot(normalized_df[CHUNK_SIZE], normalized_df[COSINE_SIMILARITY], label=COSINE_SIMILARITY)
# plt.plot(normalized_df[CHUNK_SIZE], normalized_df[CONTAIN_COUNT], label=CONTAIN_COUNT)
# plt.title("Hybrid System Comparison")
# plt.xlabel("Chunk Size")
# plt.ylabel("Value (Normalized)")
# plt.legend()
# plt.grid(True)
# plt.show()

# # 14. 정규화 안 한 결과
# plt.figure(figsize=(10, 6))
# # plt.plot(df_hybrid[CHUNK_SIZE], df_hybrid[BERT_F1], label=BERT_F1)
# plt.plot(df_hybrid[CHUNK_SIZE], df_hybrid[F1], label=F1)
# plt.plot(normalized_df[CHUNK_SIZE], normalized_df[COSINE_SIMILARITY], label=COSINE_SIMILARITY)
# # plt.plot(df_hybrid[CHUNK_SIZE], df_hybrid[CONTAIN_COUNT], label=CONTAIN_COUNT)
# plt.title("Hybrid System Comparison (Non-Normalized)")
# plt.xlabel("Chunk Size")
# plt.ylabel("Value")
# plt.legend()
# plt.grid(True)
# plt.show()
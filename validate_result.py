import pandas as pd
import os
from bert_score import score as bert_score

# 폴더 경로 (적절히 변경)
folder_path = "./results/experiment_ver3/"

# '파일명'에서 확장자를 제거하는 함수
def normalize_filename(filename):
    return os.path.splitext(filename)[0]

# 폴더 내 CSV 파일 목록을 가져와 정렬
csv_files = sorted([file for file in os.listdir(folder_path) if file.endswith(".csv")])

# 종합 결과를 저장할 리스트
summary_results = []

# 파일 이름 순서대로 반복 처리
for file_name in csv_files:
    file_path = os.path.join(folder_path, file_name)
    
    # CSV 파일 읽기
    data = pd.read_csv(file_path)
    
    # 정답(reference) 생성
    data['reference'] = data.apply(lambda row: row['answer'] + ' ' + normalize_filename(row['파일명']), axis=1)
    
    # 결과 저장용 리스트
    bert_f1s, exact_matches, f1_scores = [], [], []
    
    # Contain 여부 계산
    data['Contain'] = data.apply(lambda row: 1 if normalize_filename(row['파일명']) in row['source'] else 0, axis=1)
    contain_count = data['Contain'].sum()
    
    # 각 행에 대해 지표 계산
    for index, row in data.iterrows():
        candidate_sentence = row['result']
        reference_sentence = row['reference']
        
        # BERTScore 계산
        P, R, F1 = bert_score([candidate_sentence], [reference_sentence], lang='ko')
        bert_f1s.append(F1[0].item())
        
        # Exact Match 계산
        exact_match = 1 if candidate_sentence.strip() == reference_sentence.strip() else 0
        exact_matches.append(exact_match)
        
        # F1 Score 계산
        candidate_tokens = set(candidate_sentence.split())
        reference_tokens = set(reference_sentence.split())
        
        true_positive = len(candidate_tokens & reference_tokens)
        precision = true_positive / len(candidate_tokens) if candidate_tokens else 0
        recall = true_positive / len(reference_tokens) if reference_tokens else 0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0
        f1_scores.append(f1)
    
    # 결과를 데이터프레임에 추가
    data['bert_f1'] = bert_f1s
    data['exact_match'] = exact_matches
    data['f1_score'] = f1_scores
    
    # 평균값 계산
    avg_bert_f1 = data['bert_f1'].mean()
    avg_exact_match = data['exact_match'].mean()
    avg_f1_score = data['f1_score'].mean()
    
    # 요약 결과 저장
    summary_results.append({
        '파일명': file_name,
        '평균_BERT_F1': avg_bert_f1,
        '평균_F1_Score': avg_f1_score,
        '평균_Exact_Match': avg_exact_match,
        'Contain_Count': contain_count
    })
    
    # 처리된 파일 저장 (선택)
    processed_file_path = os.path.join(folder_path, f"processed_{file_name}")
    data.to_csv(processed_file_path, index=False)

# 요약 결과를 데이터프레임으로 저장
summary_df = pd.DataFrame(summary_results)

# 결과를 CSV 파일로 저장
summary_output_path = os.path.join(folder_path, "summary_results.csv")
summary_df.to_csv(summary_output_path, index=False)

# 요약 결과 출력
print(summary_df)

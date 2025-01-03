# BKMS2 프로젝트를 위한 실험 코드

## 프로젝트 소개
본 코드는 BKMS2 프로젝트를 위해 제작된 것으로, 특정 목표를 달성하기 위한 실험을 진행하기 위한 초기 버전입니다. 주된 목표는 다음과 같습니다:

1. 적절한 문서 파일 이름 검색
2. PDF 파일을 활용한 RAG (Retrieval-Augmented Generation)
3. 청크 및 오버랩 크기에 따른 정확도와 저장공간 비교
4. 최적의 청크 및 오버랩 크기를 설정한 후, 임베딩 모델별 정확도 및 저장공간 비교

본 코드는 위 목표들 중 **1번과 3번**에 해당하는 실험을 진행하기 위해 작성된 초보적인 코드입니다.

## 예시 결과
코드를 실행하면 다음과 같은 예시 결과를 얻을 수 있습니다:

```
...

--- Chunk Size: 300, Overlap Size: 50 ---

...

Question: OpenLnL 운영 학생 공청회는 언제 개최하나요?
Expected Answer: 2024. 11. 5.(화) 12:00 ~ 13:00
Predicted Answer: OpenLnL 운영 학생 공청회는 2024년 11월 5일 화요일, 12:00부터 13:00까지 개최됩니다. (출처: ./data/[학부&대학원] OpenLnL 운영 학생 공청회 개최.txt)

...

VectorDB Storage Size: 12.46 MB
```

## 코드 설명
본 코드는 **청크 크기(chunk size)** 와 **중첩 크기(overlap size)** 에 따른 실험을 진행하고, 쿼리에 대한 LLM 응답을 생성하는 것을 목표로 합니다. 또한, 각 설정에 따른 **벡터 저장소(VectorDB)의 저장 용량**을 출력하여 비교할 수 있도록 구현되어 있습니다.

### 주요 기능
- 텍스트 파일을 로드하고 지정된 청크 크기와 중첩 크기를 사용하여 분할합니다.
- ChromaDB를 활용해 벡터 저장소를 생성하고, 문서 청크를 저장합니다.
- 쿼리에 대해 유사한 문서를 검색하고, 해당 문서들을 기반으로 LLM이 답변을 생성합니다.

## 개선 필요 사항
아직 개선이 필요한 부분은 다음과 같습니다:

1. **출처 노출 문제**: 현재 프롬프트에서 파일 경로가 그대로 노출되고 있어, 출처를 더 적절하게 표시하는 방식으로 수정이 필요합니다. (예: `출처: ./data/~.txt)`)
2. **정확도 비교 알고리즘 미구현**: 현재는 LLM의 응답 결과를 확인하는 단계에 있으며, 정확도 비교 방식의 알고리즘은 구체적으로 구현되지 않았습니다.

이와 같은 개선 사항들을 반영하여, 앞으로 더 나은 정확도 비교와 적절한 정보 제공 방식을 구현할 예정입니다.


from sentence_transformers import SentenceTransformer, util
import torch

# 1. 모델 로드 (사용하시는 Qwen 모델 경로로 교체하세요)
# 만약 HuggingFace에 공개된 Qwen2.5-Embedding 등을 쓴다면 trust_remote_code=True가 필요할 수 있습니다.
model_id = "Qwen/Qwen3-Embedding-0.6B" # 예시 경로 (실제 Qwen3 경로로 변경)
model = SentenceTransformer(model_id, trust_remote_code=True)

# 2. 분류할 클래스 정의 및 임베딩 생성 (미리 벡터화)
labels = ["예약", "결제", "주문", "주식"]
label_embeddings = model.encode(labels, convert_to_tensor=True)

# 3. 입력 문장
query = "삼성전자 10주 매수해줘"

# 4. 입력 문장 임베딩 생성
query_embedding = model.encode(query, convert_to_tensor=True)

# 5. 코사인 유사도 계산
cos_scores = util.cos_sim(query_embedding, label_embeddings)[0]

# 6. 가장 높은 점수의 인덱스 찾기
best_score_idx = torch.argmax(cos_scores).item()
predicted_label = labels[best_score_idx]

print(f"입력: {query}")
print(f"예측 클래스: {predicted_label}")
print(f"유사도 점수: {cos_scores}")

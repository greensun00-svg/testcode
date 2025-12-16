from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import numpy as np

# 1. 모델 로드
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)

# 2. 학습 데이터 준비 (예시)
train_sentences = [
    "내일 7시에 식당 잡아줘", "호텔 예약 확인해줘",       # 예약
    "카드로 계산할게요", "결제 내역 보여줘",              # 결제
    "아이스 아메리카노 한 잔", "치킨 배달 시켜줘",        # 주문
    "테슬라 주가 어때?", "애플 주식 사줘"                 # 주식
]
train_labels = ["예약", "예약", "결제", "결제", "주문", "주문", "주식", "주식"]

# 3. 문장을 벡터(X)로 변환
X_train = model.encode(train_sentences)
y_train = train_labels

# 4. 가벼운 분류기 학습 (Logistic Regression)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# --- 실전 테스트 ---
test_text = "이번 달 카드값 얼마야?"
test_vector = model.encode([test_text]) # 2차원 배열 형태로 입력

prediction = classifier.predict(test_vector)
print(f"입력: {test_text} -> 예측: {prediction[0]}")

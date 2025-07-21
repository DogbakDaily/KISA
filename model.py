import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
from torch.utils.data import Dataset, DataLoader

# --- 1. 데이터 로드 ---
try:
    df = pd.read_csv('spam_data.csv')
except FileNotFoundError:
    print("spam_data.csv 파일을 찾을 수 없습니다. 예시 데이터를 생성합니다.")
    data = {
        'text': [
            "Free money! Click here now!",
            "Hi, how are you doing today?",
            "You've won a prize! Reply to claim.",
            "Meeting at 3 PM tomorrow.",
            "Urgent: Your account will be locked if not updated. Visit link.",
            "Let's grab coffee this week.",
            "Exclusive offer: Get 50% off all products for a limited time!",
            "Did you get my email about the project?",
            "Your package delivery failed. Please update your shipping info at [link]",
            "Just confirming our dinner plans for Friday."
        ],
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)

texts = df['text'].tolist()
labels = df['label'].tolist()

# --- 2. 훈련/검증/테스트 세트 분리 ---
# 먼저 훈련 세트와 (검증 + 테스트) 세트로 분리
train_texts, val_test_texts, train_labels, val_test_labels = train_test_split(
    texts, labels, test_size=0.3, random_state=42, stratify=labels
)
# 이후 (검증 + 테스트) 세트를 검증 세트와 테스트 세트로 분리
val_texts, test_texts, val_labels, test_labels = train_test_split(val_test_texts, val_test_labels, test_size=0.5, random_state=42, stratify=val_test_labels)

print(f"훈련 세트: {len(train_texts)}개")
print(f"검증 세트: {len(val_texts)}개")
print(f"테스트 세트: {len(test_texts)}개")

# --- 3. 토크나이저 로드 및 데이터 토큰화 ---
# 'distilbert-base-uncased'는 영어 소문자 전용 모델
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# 각 데이터셋 토큰화
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# --- 4. PyTorch Dataset 생성 ---
class SpamDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # encodings는 딕셔너리 형태이므로 각 키에 접근하여 PyTorch 텐서로 변환
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]) # 레이블 추가
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SpamDataset(train_encodings, train_labels)
val_dataset = SpamDataset(val_encodings, val_labels)
test_dataset = SpamDataset(test_encodings, test_labels)

print("\n데이터셋 생성 완료.")
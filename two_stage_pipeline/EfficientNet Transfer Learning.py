import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os
from torch.amp import autocast, GradScaler

# ==========================================
# 1. 하이퍼파라미터 및 설정 (이 부분을 덮어쓰세요)
# ==========================================
BATCH_SIZE = 16 
EPOCHS = 20
LEARNING_RATE = 0.001

# [핵심] 변수명은 반드시 DATASET_DIR 이어야 하며, STEP 1에서 만든 안전한 폴더를 가리켜야 합니다.
DATASET_DIR = r"E:\Robot_Team_Project\EfficientNet_Dataset\Safe_Split_Dataset"

NUM_CLASSES = 4

# GPU 사용 가능 여부 확인
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"학습에 사용되는 기기: {device}")

# 2. 데이터 전처리 파이프라인 (Transforms)
# 날씨, 햇빛, 그림자 변화에 강한 모델을 만들기 위해 '강력한 증강'을 추가합니다.
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        
        # --- [날씨 및 조명 대응 강력 증강 세트] ---
        # 1. 좌우 반전 (부품의 방향에 상관없이 학습)
        transforms.RandomHorizontalFlip(p=0.5),
        
        # 2. 회전 (OBB가 완벽하지 않을 때를 대비해 미세 회전)
        transforms.RandomRotation(15),
        
        # 3. ★핵심: 밝기, 대비, 채도, 색상 변형 (태양광 및 조명 변화 대응)
        # brightness(0.5): 밝기를 대폭 조절, contrast(0.5): 그림자와 반사광 대비
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.1),
        
        # 4. 랜덤 그레이스케일 (색상보다 질감에 집중하게 만듦)
        transforms.RandomGrayscale(p=0.1),
        
        # 5. 랜덤 가우시안 블러 (카메라 초점이 미세하게 나갔을 때 대비)
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        # ------------------------------------------
        
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 3. 데이터셋 및 데이터 로더 준비
image_datasets = {
    x: datasets.ImageFolder(os.path.join(DATASET_DIR, x), data_transforms[x])
    for x in ['train', 'val']
}

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == 'train'), num_workers=4)
    for x in ['train', 'val']
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(f"클래스 종류: {class_names}")

# 4. EfficientNet 전이학습 모델 설계
# 사전 학습된 가중치가 포함된 EfficientNet-B0 로드
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

# [핵심] 기존 신경망 층 동결 (가중치 업데이트 방지)
for param in model.parameters():
    param.requires_grad = False

# 모델의 마지막 분류기(Classifier) 교체
# EfficientNet-b0의 마지막 계층을 현재 프로젝트의 클래스 개수(NUM_CLASSES)에 맞게 재정의합니다.
num_ftrs = model.classifier[1].in_features
# 새롭게 추가된 이 계층의 파라미터는 requires_grad=True 가 기본값으로 설정됩니다.
model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)

model = model.to(device)

# 5. 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
# 동결되지 않은 계층(model.classifier)의 파라미터만 학습되도록 옵티마이저에 전달합니다.
optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

# 6. 모델 학습 루프
def train_model(model, criterion, optimizer, num_epochs=10):
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 각 에포크마다 학습 단계와 검증 단계를 거침
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 학습 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터 반복
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() # 기울기 초기화

                # 순전파
                # 학습 시에만 연산 기록을 추적
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 학습 단계인 경우 역전파 및 최적화
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 통계 계산
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 검증 단계에서 성능이 향상되었을 때 가중치 복사
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'best_effnet_weights.pth')
                
    print(f'학습 완료! 검증 세트 최고 정확도: {best_acc:4f}')
    return model

# 7. 학습 실행
if __name__ == '__main__':
    model = train_model(model, criterion, optimizer, num_epochs=EPOCHS)
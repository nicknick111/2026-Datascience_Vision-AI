import torch
from ultralytics import YOLO

# 1. 원래 모델 경로
old_path = r"E:\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\runs\detect\runs_rep\rep_train3\weights\best.pt"
# 2. 새로 저장할 경로
new_path = r"E:\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\runs\detect\runs_rep\rep_train3\weights\best_fixed.pt"

try:
    print("모델 가중치 추출 중...")
    # 가중치만 불러오기 (문제가 되는 메타데이터는 무시)
    ckpt = torch.load(old_path, map_location='cpu')
    
    # 3. 새로운 YOLO 객체 생성 후 가중치만 덮어쓰기
    model = YOLO("yolov26n.pt") # 기본 구조 로드 (yolo26n이면 해당 모델)
    model.load(old_path)       # 가중치 이식
    
    # 4. 현재 환경(NumPy 1.26.4)에 맞게 새로 저장
    model.save(new_path)
    print(f"✅ 모델 세척 완료! 새 파일: {new_path}")
    
except Exception as e:
    print(f"❌ 세척 실패: {e}")
    print("💡 이 경우, 잠시 NumPy 2.x로 올려서 모델을 불러온 뒤 'ONNX'로 변환하는 것이 가장 확실합니다.")
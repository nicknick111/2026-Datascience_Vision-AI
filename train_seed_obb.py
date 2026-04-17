# train_seed_obb.py
from ultralytics import YOLO

def main():
    # 1. YOLOv8 Nano OBB 사전 학습 모델 로드
    # 일반 yolov8n.pt가 아닌 obb 전용 모델을 사용해야 합니다.
    model = YOLO('yolov8n-obb.pt')

    # 2. yaml 파일 경로 지정 (절대 경로 사용 권장)
    yaml_path = r'E:\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\Dataset_OBB\data.yaml'

    # 3. 모델 학습 시작 (Seed 모델이므로 epoch는 50~100 정도가 적당합니다)
    print("🚀 YOLOv8-OBB Seed Model 학습을 시작합니다...")
    results = model.train(
        data=yaml_path,
        epochs=50,          # 학습 반복 횟수
        imgsz=640,          # 이미지 크기
        batch=16,           # PC의 VRAM에 따라 조절 (메모리 부족 시 8로 낮춤)
        device=0,           # 0: GPU 사용, 'cpu': CPU 사용
        name='seed_obb_v1'  # 결과가 저장될 폴더 이름 (runs/obb/seed_obb_v1)
    )
    print("✅ 학습 완료! 결과는 runs/obb/seed_obb_v1 폴더에 저장되었습니다.")

if __name__ == '__main__':
    main()
from ultralytics import YOLO

def main():
    # 1. YOLOv8 Nano OBB 기본 모델 로드 (가장 가볍고 빠른 모델)
    model = YOLO('yolov8n-obb.pt')
    
    print("🚀 [Phase 1] YOLOv8n-OBB 최종 학습을 시작합니다...")
    
    # 2. yaml 파일 절대 경로 지정
    yaml_path = r'E:\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\dataset_final.yaml'

    # 3. 모델 학습 (Training)
    results = model.train(
        data=yaml_path,
        epochs=150,          # 충분한 특징 학습을 위해 150 에포크 지정
        imgsz=640,           # 기본 해상도 유지
        batch=8,             # ⚠️ GTX 1050 2GB VRAM 초과 방지를 위해 Batch Size 8로 제한
        device=0,            # 로컬 GPU 사용
        
        # [산업 현장 맞춤형 데이터 증강 (Augmentation)]
        degrees=180.0,       # (핵심) 부품이 어떤 각도로 놓여 있어도 인식하도록 180도 회전
        flipud=0.5,          # 상하 반전 (탑다운 뷰 카메라 환경에 대응)
        mosaic=1.0,          # 4장의 이미지를 섞어 배경 과적합(컨베이어 벨트 무늬 암기) 원천 차단
        hsv_s=1.0,      # 대표님 제안 수치 (매우 좋음)
        hsv_v=0.9,      # 대표님 제안 수치 (매우 좋음)
        perspective=0.001, # 미세한 각도 왜곡 추가
        box=10.0,       # 박스 위치 정확도 가중치 강화 (OBB 정밀도 향상)
        cls=2.0         # 클래스 분류 가중치 강화 (C4/Defect 구분력 향상)
        )
        
    name='final_obb_extractor' # 결과물이 저장될 폴더명
    
    
    print("✅ 최종 학습 완료! 가장 성능이 좋은 모델은 'runs/obb/final_obb_extractor/weights/best.pt'에 저장됩니다.")

if __name__ == '__main__':
    main()
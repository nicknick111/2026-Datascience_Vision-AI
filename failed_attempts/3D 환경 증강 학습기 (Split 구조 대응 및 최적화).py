import os
import yaml
from ultralytics import YOLO

def setup_dataset_and_train_final():
    """
    [최종 3단계: YOLO 모델 실전 학습]
    수동 및 자동 라벨링, 검수까지 모두 완료된 전체 데이터셋을 바탕으로
    실제 검사기(금속 그리퍼 고정) 환경과 조명 변화에 강인한 최종 AI를 학습시킵니다.
    """
    current_working_dir = os.getcwd()
    # 경로 인식 오류를 방지하기 위해 절대 경로로 지정하는 것을 권장합니다.
    # 만약 dataset 폴더가 다른 곳에 있다면 아래 경로를 수정해 주세요.
    base_dir = os.path.join(current_working_dir, "dataset")
    
    # 학습 시작 전 라벨링 파일 존재 여부 안전 검사
    labels_train_dir = os.path.join(base_dir, "labels", "train")
    if not os.path.exists(labels_train_dir) or len(os.listdir(labels_train_dir)) == 0:
        print("❌ 오류: 'dataset/labels/train' 폴더에 라벨링 파일(.txt)이 없습니다.")
        return

    print("📂 데이터셋 구조를 확인했습니다. data.yaml을 최종 버전으로 업데이트합니다...")

    # 💡 [수정 1] 최종 목표 클래스 3가지로 변경
    classes = ["inspection_C4_Bomb", "inspection_Communication", "inspection_Supply"]
    yaml_path = os.path.join(base_dir, "data.yaml")
    
    # 💡 [수정 2] val 폴더를 정상적으로 검증용으로 사용하도록 매핑
    yaml_data = {
        "path": base_dir,
        "train": "images/train",  
        "val": "images/val",      # 이전 임시 학습 때 train으로 막아둔 것을 val로 원상복구
        "test": "images/test",    
        "nc": len(classes),
        "names": classes
    }
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False)
        
    print(f"✅ 설정 파일 생성 완료: {yaml_path}")
    
    # YOLOv8 Nano 모델 로드 (v1 모델에서 이어서 하지 않고, 깨끗한 모델에서 처음부터 학습)
    model = YOLO("yolov8n.pt") 
    print("\n🚀 실전 투입용 최종 AI 모델(final_inspection_model) 학습을 시작합니다...")

    # 💡 [수정 3] 본격적인 학습 진행 (그리퍼 검사기 환경 맞춤형 파라미터)
    results = model.train(
        data=yaml_path,
        epochs=150,             # 전체 데이터가 준비되었으므로 충분히 반복 학습
        imgsz=640,              
        batch=16,               # GPU 메모리가 부족하면 8로 낮추세요
        project="Drone_QC_Project",
        name="final_inspection_model", # 최종 모델 저장 폴더명 변경
        
        # ─── 물리 환경 및 공장 조명 시뮬레이션 파라미터 (최적화) ───
        # 기존 드론 비행용(과도한 3D 왜곡) 세팅을 버리고, 고정식 검사기에 맞게 튜닝했습니다.
        degrees=5.0,        # 미세한 기계 진동 및 회전 오차만 허용 (45도 -> 5도)
        translate=0.1,      # 위치 이동 최소화
        scale=0.2,          # 크기 변화 최소화
        shear=2.0,          # 찌그러짐 왜곡 최소화 (15.0 -> 2.0)
        perspective=0.0,    # 카메라가 고정되어 있으므로 원근법(투시) 왜곡 제거
        flipud=0.0,         # 로봇이 물체를 거꾸로 쥘 일은 없으므로 상하 반전 제거
        fliplr=0.0,         # 좌우 반전 제거
        mosaic=0.5,         # 부분 가림 학습 (1.0 -> 0.5로 하향)
        
        # 조명(과노출 및 어두움) 대비 강화
        hsv_h=0.015,        # 조명 색온도 변화
        hsv_s=0.5,          # 그림자/반사 변화
        hsv_v=0.6           # 💡 어두운 C4 인식을 위해 밝기 변화폭을 높게(0.6) 설정하여 강인함 확보
    )
    
    print("\n🎉 최종 학습 완료! 최고 성능 모델이 'Drone_QC_Project/final_inspection_model/weights/best.pt'에 저장되었습니다.")

if __name__ == "__main__":
    setup_dataset_and_train_final()
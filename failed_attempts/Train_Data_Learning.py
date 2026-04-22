import os
import yaml
from ultralytics import YOLO

def setup_dataset_and_train():
    current_working_dir = os.getcwd()
    base_dir = os.path.join(current_working_dir, "dataset")
    
    # 1. 클래스명 수정 (Classes.txt와 완벽히 일치해야 함)
    classes = ["inspection_C4_Bomb", "inspection_Communication", "inspection_Supply"]
    yaml_path = os.path.join(base_dir, "data.yaml")
    
    # 2. Val 에러 방지: 첫 학습이므로 val 경로를 임시로 train으로 돌려둠
    yaml_data = {
        "path": base_dir,
        "train": "images/train",  
        "val": "images/train",  # 임시 조치: 라벨이 없는 val 폴더 검사 우회
        "test": "images/test",    
        "nc": len(classes),
        "names": classes
    }
    
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False)
        
    model = YOLO("yolov8n.pt") 

    # 3. 그리퍼(검사기) 환경에 맞춘 증강 파라미터 최적화 (회전/왜곡 최소화, 조명 위주)
    results = model.train(
        data=yaml_path,
        epochs=50,          # 자동 라벨링용 초벌 학습이므로 50으로 단축
        imgsz=640,              
        batch=16, 
        project="Drone_QC_Project",
        name="v1_auto_label_model", # 폴더명 변경
        degrees=5.0,        # 그리퍼 고정이므로 미세한 떨림만 허용
        translate=0.1,      
        scale=0.2,          
        shear=2.0,         
        perspective=0.0,    # 원근 왜곡 제거 (고정 카메라)
        flipud=0.0,         # 상하 반전 제거 (항상 같은 방향으로 파지됨)
        fliplr=0.0,         
        mosaic=0.5,         
        hsv_h=0.015,        
        hsv_s=0.5,          
        hsv_v=0.4           # 어두운 객체 탐지를 위해 밝기 변화는 유지
    )

if __name__ == "__main__":
    setup_dataset_and_train()
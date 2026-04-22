import cv2                  # [컴퓨터 비전] 영상 파일 열기, 프레임 자르기, 해상도 조절(Resize)을 위한 라이브러리
import os                   # [운영체제] 폴더 생성, 파일 경로/개수 확인 등을 처리하는 라이브러리
import yaml                 # [환경설정] YOLOv8 학습에 필요한 설정 파일(data.yaml)을 만들고 읽기 위한 라이브러리
import numpy as np          # [수학/배열] 윈도우 한글 경로 이미지 저장 오류를 완벽하게 해결하기 위한 라이브러리
from datetime import datetime # [시간] 이미지 파일명 생성 시 현재 시간을 스탬프 찍기 위해 사용
from ultralytics import YOLO  # [인공지능] 최신 YOLOv8 모델을 불러오고 학습(Train)시키기 위한 핵심 라이브러리

# ==========================================
# [설정] 오직 이것만 인식하게 됩니다! (사용자 정의 학습 분야)
# 설명: 여기에 적힌 17개의 클래스 외에 기존 YOLO가 알던 사람, 자동차 등은 
#       학습 과정에서 완전히 덮어씌워져(지워져) 인식되지 않게 됩니다.
# ==========================================
CLASS_NAMES = [
    "UAV_Wheel", "Left_wing", "Right_wing", "Fuselage_Part_1", "Fuselage_Part_2_1", 
    "Fuselage_Part_2_2", "Fuselage_Part_3_1", "Fuselage_Part_3_2", "Fuselage_Part_4_1", 
    "ISR_System_1", "ISR_System_2", "Jet_Engine_1", "Jet_Engine_2", 
    "Missile_Launcher", "Missile", "Pylon", "UAV_Fuselage_1"
]

# ==========================================
# [기능 1] 영상 추출, 전처리 및 대표 데이터 분리 저장
# ==========================================
def extract_and_preprocess(video_path, class_id, class_name, step=5):
    """
    동영상을 캡처하여 이미지를 만들 때, 전체 데이터는 train 폴더에 넣고,
    일부 대표 데이터는 별도의 라벨링 전용 폴더에 따로 저장합니다.
    """
    if not os.path.exists(video_path):
        print(f"  [-] 누락: '{video_path}' 파일을 찾을 수 없습니다.")
        return

    # 1. 전체 데이터가 저장될 메인 폴더
    train_dir = "datasets/images/train"
    # 2. [추가된 목표] 대표 데이터만 따로 모아둘 라벨링 전용 폴더
    sample_dir = "datasets/images/representative_samples"
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    saved = 0
    sample_saved = 0

    print(f"\n▶ [{class_name}] 초고속 영상 추출 및 전처리 시작...")
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            break # 영상이 끝나면 반복문 종료

        # 모델 학습에 적합하도록 이미지 크기 640x640으로 조절 (전처리)
        frame_resized = cv2.resize(frame, (640, 640))
        
        # 파일 이름에 시간과 번호를 붙여 겹치지 않게 생성
        base_filename = f"{class_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{saved:04d}.jpg"
        train_filepath = os.path.join(train_dir, base_filename)
        
        # [해결] 한글 경로 저장 오류를 방지하기 위해 numpy 배열 변환 후 강제 쓰기 방식 사용
        is_success, img_arr = cv2.imencode('.jpg', frame_resized)
        if is_success:
            # 기본적으로 모든 추출된 이미지는 메인 학습 폴더에 저장합니다.
            with open(train_filepath, 'wb') as f:
                img_arr.tofile(f)
            
            # [목표 달성] 10장 저장될 때마다 1장씩 대표 이미지로 선정하여 별도 폴더에 추가 복사 저장!
            if saved % 10 == 0:
                sample_filepath = os.path.join(sample_dir, f"SAMPLE_{base_filename}")
                with open(sample_filepath, 'wb') as f:
                    img_arr.tofile(f)
                sample_saved += 1
        
        saved += 1
        print(f"  - 전체 {saved:04d}장 추출 (대표 데이터 {sample_saved}장 분리 중...)", end="\r")

        # 설정한 step(기본 5프레임)만큼 건너뛰며 너무 비슷한 사진이 연속으로 찍히는 것을 방지
        for _ in range(step - 1):
            if not cap.grab():
                break

    cap.release()
    print(f"\n  ✅ [{class_name}] 총 {saved}장 이미지 추출 완료. (대표 이미지 {sample_saved}장 따로 보관됨)")


# ==========================================
# [기능 2] YOLOv8n AI Model 학습 (기존 지식 지우고 새 지식 주입)
# ==========================================
def train_yolo_model():
    """
    이 함수가 실행되면, 제공된 17개의 클래스만 알도록 AI의 두뇌 구조가 재편성됩니다.
    기존에 알던 사물들은 이 과정에서 잊혀지며, 오직 새 클래스만 인식하게 됩니다.
    """
    train_dir = "datasets/images/train"
    
    if not os.path.exists(train_dir):
        print("🔴 학습할 데이터 폴더가 없습니다. 영상을 먼저 추출해주세요.")
        return
        
    total_data_count = len([f for f in os.listdir(train_dir) if f.endswith('.jpg') or f.endswith('.png')])
    
    if total_data_count == 0:
        print("🔴 폴더 내에 학습할 이미지 데이터가 확인되지 않습니다.")
        return
        
    print(f"\n🚀 폴더 내 확인된 데이터 총 {total_data_count:,}장으로 커스텀 학습을 시작합니다!")
    
    # 1. YOLO 학습을 위한 환경설정 파일(yaml) 생성
    # 이 파일이 기존의 COCO 데이터셋을 대체하여 '우리의 분야'만 인식하도록 만드는 핵심입니다.
    yaml_path = os.path.abspath("datasets/drone_parts.yaml")
    data_config = {
        "path": os.path.abspath("datasets"),
        "train": "images/train",  
        "val": "images/train",    
        "nc": len(CLASS_NAMES),   # 17개 클래스
        "names": CLASS_NAMES      # 17개 클래스 이름
    }
    
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, allow_unicode=True, sort_keys=False)
    
    # 2. 가장 가볍고 빠른 YOLOv8n 모델 불러오기
    model = YOLO("yolov8n.pt")
    print("\n[INFO] 기존의 사물 인식 지식을 지우고, 무인기 부품만을 위한 새로운 뇌 신경망을 구성합니다...")
    
    # 3. 본격적인 학습 시작 (기존 분야 제외 처리됨)
    model.train(
        data=yaml_path,
        epochs=100,        
        imgsz=640,         
        batch=16,          
        patience=20,       
        degrees=45.0,      
        perspective=0.001, 
        flipud=0.3,        
        fliplr=0.5,        
        project="runs/train",        
        name="drone_custom_model",
        device=0  
    )
    print("\n🎉 [완료] 오직 사용자 정의 분야만 인식하는 인공지능이 탄생했습니다!")
    print("👉 완성된 모델 파일 위치: 'runs/train/drone_custom_model/weights/best.pt'")


# ==========================================
# 메인 통합 컨트롤러 (프로그램 실행 진입점)
# ==========================================
if __name__ == "__main__":
    
    print("\n" + "=" * 60)
    ans = input("❓ [1단계] 영상에서 이미지를 추출하고 대표 데이터를 따로 분리할까요? (y/n): ")
    if ans.lower() == 'y':  
        # 추출할 17개의 모든 비디오 목록 복구 완료
        video_datasets = [
            {"video_path": "UAV_Wheel.mp4", "class_id": 0, "class_name": "UAV_Wheel"},
            {"video_path": "20260323_Left_wing.mp4", "class_id": 1, "class_name": "Left_wing"},
            {"video_path": "20260323_Right_wing.mp4", "class_id": 2, "class_name": "Right_wing"},
            {"video_path": "Internal_Assembly_of_Model_UAV_Fuselage_Part_1.mp4", "class_id": 3, "class_name": "Fuselage_Part_1"},
            {"video_path": "Internal_Assembly_of_Model_UAV_Fuselage_Part_2_1.mp4", "class_id": 4, "class_name": "Fuselage_Part_2_1"},
            {"video_path": "Internal_Assembly_of_Model_UAV_Fuselage_Part_2_2.mp4", "class_id": 5, "class_name": "Fuselage_Part_2_2"},
            {"video_path": "Internal_Assembly_of_Model_UAV_Fuselage_Part_3_1.mp4", "class_id": 6, "class_name": "Fuselage_Part_3_1"},
            {"video_path": "Internal_Assembly_of_Model_UAV_Fuselage_Part_3_2.mp4", "class_id": 7, "class_name": "Fuselage_Part_3_2"},
            {"video_path": "Internal_Assembly_of_Model_UAV_Fuselage_Part_4_1.mp4", "class_id": 8, "class_name": "Fuselage_Part_4_1"},
            {"video_path": "ISR_System_part_1.mp4", "class_id": 9, "class_name": "ISR_System_1"},
            {"video_path": "ISR_System_part_2.mp4", "class_id": 10, "class_name": "ISR_System_2"},
            {"video_path": "jet_engine_part_1.mp4", "class_id": 11, "class_name": "Jet_Engine_1"},
            {"video_path": "jet_engine_part_2.mp4", "class_id": 12, "class_name": "Jet_Engine_2"},
            {"video_path": "Missile_Launcher.mp4", "class_id": 13, "class_name": "Missile_Launcher"},
            {"video_path": "Missile.mp4", "class_id": 14, "class_name": "Missile"},
            {"video_path": "Pylon.mp4", "class_id": 15, "class_name": "Pylon"},
            {"video_path": "Uav_Fuselage_part_1.mp4", "class_id": 16, "class_name": "UAV_Fuselage_1"}
        ]
        
        for data in video_datasets:
            extract_and_preprocess(data["video_path"], data["class_id"], data["class_name"])
            
    print("\n" + "=" * 60)
    print("⚠️ 필수 안내: 인공지능을 학습시키려면 이미지 외에도 박스가 그려진 '라벨링 파일(.txt)'이 반드시 필요합니다.")
    print("💡 꿀팁: 너무 막막하다면 'datasets/images/representative_samples' 폴더에 모인")
    print("         대표 데이터들만 먼저 라벨링해서 학습을 돌려보는 것을 추천합니다!")
    print("=" * 60)
    
    ans = input("❓ [2단계] 생성된 사용자 정의 분야로 AI 모델 커스텀 학습을 시작하시겠습니까? (y/n): ")
    if ans.lower() == 'y':
        train_yolo_model()
    else:
        print("작업을 종료합니다. 언제든 라벨링이 끝나면 다시 실행하여 학습을 시작하세요!")
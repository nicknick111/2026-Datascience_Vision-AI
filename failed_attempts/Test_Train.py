import cv2          # [영상 처리] 동영상을 읽고 이미지를 자르거나 크기를 줄이는 데 사용하는 라이브러리입니다.
import os           # [파일/폴더 관리] 컴퓨터의 폴더를 만들거나 파일이 있는지 확인하는 역할을 합니다.
import yaml         # [설정 파일 생성] 인공지능이 학습할 때 참고할 '데이터 지도(.yaml)'를 만드는 데 쓰입니다.
import random       # [랜덤 뽑기] 수많은 사진 중 무작위로 Train(학습용)과 Val(검증용/모의고사)을 나누기 위해 사용합니다.
import numpy as np  # [수학 계산] 이미지 데이터를 컴퓨터가 이해할 수 있는 숫자로 변환할 때 보조해줍니다.
import pymysql      # [데이터베이스] 추출된 데이터와 학습 기록을 MySQL DB에 꼼꼼히 적어두는 '기록장' 역할을 합니다.
from datetime import datetime  # [시간 기록] 파일 이름에 현재 시간을 넣거나 DB에 저장 시간을 기록할 때 씁니다.
from ultralytics import YOLO   # [AI 핵심] 제품 상태를 인식하게 될 YOLO 인공지능 모델을 불러오는 핵심 라이브러리입니다.

# ==========================================
# [변경 1] 인공지능이 외워야 할 '정답지 목록' (정상 vs 파손)
# ==========================================
# 기존 17개 드론 부품에서 딱 2가지(정상 제품, 파손 제품)로 줄였습니다.
# 0번: Product_Normal (정상) / 1번: Product_Break (파손)
CLASS_NAMES = [
    "Product_Normal", 
    "Product_Break"
]

# ==========================================
# [변경 2] DB 연결 정보 (제품 검사 전용으로 새로 만듭니다!)
# ==========================================
DB_CONFIG = {
    'host': 'localhost',      # DB가 설치된 컴퓨터 주소
    'user': 'root',           # DB 관리자 아이디
    'password': '1234',       # DB 관리자 비밀번호 (본인 환경에 맞게 꼭 수정하세요)
    'charset': 'utf8mb4'      # 한글 깨짐 방지
}
# 기존 'drone_vision_db'와 섞이지 않게 'product_inspection_db'라는 새로운 금고를 씁니다.
DB_NAME = 'product_inspection_db'   

# ==========================================
# [변경 3] 완전히 새로운 이미지 저장 폴더 지정 (기존 datasets 폴더와 완벽 분리)
# ==========================================
# 기존에 사용하시던 'E:\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\datasets' 
# 경로와 섞이지 않도록, 현재 코드가 실행되는 위치에 완전히 새로운 이름의 폴더를 생성합니다.
# 만약 아예 다른 드라이브(예: D:\My_New_Data)에 저장하고 싶으시다면 아래 글자를 "D:/My_New_Data"로 바꾸시면 됩니다.
BASE_DATA_DIR = "Product_Inspection_Data"

def init_db():
    """
    [기능 0] 데이터베이스 준비하기
    프로그램이 켜지면 제일 먼저 실행되며, MySQL에 새로운 'product_inspection_db'를 만듭니다.
    """
    try:
        # 1. 데이터베이스(금고) 자체를 생성합니다.
        conn = pymysql.connect(host=DB_CONFIG['host'], user=DB_CONFIG['user'], password=DB_CONFIG['password'], charset=DB_CONFIG['charset'])
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        conn.commit()
        conn.close()

        # 2. 방금 만든 데이터베이스 안으로 들어가서 2개의 테이블(표)을 만듭니다.
        conn = pymysql.connect(
            host=DB_CONFIG['host'], user=DB_CONFIG['user'], password=DB_CONFIG['password'],
            database=DB_NAME, charset=DB_CONFIG['charset'], cursorclass=pymysql.cursors.DictCursor
        )
        cursor = conn.cursor()
        
        # [테이블 A] 추출한 제품 사진의 정보를 기록하는 표
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extracted_images (
                id INT AUTO_INCREMENT PRIMARY KEY, -- 번호표 (자동 부여)
                class_name VARCHAR(50),            -- 제품 상태 (예: Product_Normal)
                file_name VARCHAR(255),            -- 사진 파일 이름
                dataset_type VARCHAR(10),          -- 학습용(train)인지 모의고사용(val)인지 구분
                extracted_at DATETIME              -- 사진을 뽑아낸 정확한 시간
            )
        """)
        
        # [테이블 B] AI 학습 완료 후 기록을 남기는 표
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                model_name VARCHAR(100),           -- 만들어진 AI 모델의 이름
                weights_path VARCHAR(255),         -- AI 모델이 저장된 위치
                trained_at DATETIME                -- 학습이 완료된 시간
            )
        """)
        conn.commit()
        conn.close()
        print("[DB] 제품 검사용 MySQL 데이터베이스 초기화 및 연결 성공! (기록 준비 완료)")
    except Exception as e:
        print(f"[DB ERROR] MySQL 연결 실패. 비밀번호나 서버 상태를 확인하세요: {e}")

# ==========================================
# [기능 1] 영상에서 사진 뽑아내기 (추출량 대폭 증가 버전)
# ==========================================
# 영상이 몇 초 안 되기 때문에, step 값을 기본 15에서 3으로 줄였습니다.
# step=3 의 의미: 영상의 프레임(장면)을 3장마다 1장씩 캡처합니다. (약 0.1초당 1장 추출)
# 만약 영상이 10초(약 300프레임)라면, 기존 20장에서 무려 100장의 사진을 추출하게 됩니다!
def extract_and_preprocess(video_path, class_id, class_name, step=3):
    if not os.path.exists(video_path):
        print(f"  [-] 누락: '{video_path}' 파일을 찾을 수 없습니다. 영상을 같은 폴더에 넣어주세요.")
        return

    # [수정] 방금 위에서 만든 완전히 새로운 폴더(BASE_DATA_DIR) 안에 하위 폴더들을 만듭니다.
    train_dir = os.path.join(BASE_DATA_DIR, "images/train")
    val_dir = os.path.join(BASE_DATA_DIR, "images/val")
    sample_dir = os.path.join(BASE_DATA_DIR, "images/representative_samples")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    estimated_total_extracts = total_frames // step
    sample_interval = max(1, estimated_total_extracts // 30)
    
    saved_train = 0
    saved_val = 0
    sample_saved = 0
    extract_count = 0 
    db_records = []

    # 전체 예상 추출 장수를 화면에 미리 보여줍니다.
    print(f"\n▶ [{class_name}] 상태 영상 추출 시작 (예상 추출량: 약 {estimated_total_extracts}장) ...")
    
    while True:
        ret, frame = cap.read()
        if not ret: 
            break

        # 학습 속도를 위해 320x320으로 리사이징
        frame_resized = cv2.resize(frame, (320, 320))
        
        now_time = datetime.now()
        base_filename = f"{class_name}_{now_time.strftime('%Y%m%d%H%M%S')}_{extract_count:04d}.jpg"
        
        is_val = random.random() < 0.2 
        dataset_type = "val" if is_val else "train"
        target_dir = val_dir if is_val else train_dir
        target_filepath = os.path.join(target_dir, base_filename)
        
        is_success, img_arr = cv2.imencode('.jpg', frame_resized)
        if is_success:
            with open(target_filepath, 'wb') as f:
                img_arr.tofile(f)
            
            if is_val: saved_val += 1
            else: saved_train += 1

            db_records.append((class_name, base_filename, dataset_type, now_time.strftime('%Y-%m-%d %H:%M:%S')))
            
            # 대표 샘플 30장 추출 로직 (라벨링용)
            if extract_count % sample_interval == 0 and sample_saved < 30:
                sample_filename = f"SAMPLE_{class_name}_{sample_saved+1:02d}.jpg"
                sample_filepath = os.path.join(sample_dir, sample_filename)
                with open(sample_filepath, 'wb') as f:
                    img_arr.tofile(f)
                sample_saved += 1
        
        extract_count += 1
        print(f"  - 진행: 공부용 {saved_train}장 / 모의고사용 {saved_val}장 (대표 샘플 {sample_saved}/30 분리 중...)", end="\r")

        # [중요] 영상 프레임 건너뛰기
        # 짧은 영상에서 더 많은 데이터를 얻기 위해 step 값(기본 3) 만큼만 건너뜁니다.
        # 만약 더 촘촘하게(모든 장면을 다) 뽑고 싶다면 아래 코드를 실행할 때 step=1 로 주시면 됩니다.
        for _ in range(step - 1):
            if not cap.grab(): break

    cap.release()
    print(f"\n  ✅ [{class_name}] 총 {saved_train + saved_val}장 추출 완료!")

    # 추출된 정보 DB 일괄 저장
    if db_records:
        try:
            conn = pymysql.connect(host=DB_CONFIG['host'], user=DB_CONFIG['user'], password=DB_CONFIG['password'], database=DB_NAME, charset=DB_CONFIG['charset'])
            cursor = conn.cursor()
            sql = "INSERT INTO extracted_images (class_name, file_name, dataset_type, extracted_at) VALUES (%s, %s, %s, %s)"
            cursor.executemany(sql, db_records)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"  [DB ERROR] 데이터베이스 저장 실패: {e}")

# ==========================================
# [기능 2] 제품 상태 판독 인공지능 학습 시작
# ==========================================
def train_yolo_model():
    # [수정] AI가 학습할 때도 새롭게 지정된 폴더(BASE_DATA_DIR)에서 사진을 찾아옵니다.
    train_dir = os.path.join(BASE_DATA_DIR, "images/train")
    val_dir = os.path.join(BASE_DATA_DIR, "images/val")
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("🔴 폴더가 없습니다. [1단계] 영상을 먼저 추출해주세요.")
        return
        
    train_count = len([f for f in os.listdir(train_dir) if f.endswith('.jpg')])
    val_count = len([f for f in os.listdir(val_dir) if f.endswith('.jpg')])
    
    if train_count == 0 or val_count == 0:
        print("🔴 폴더 내에 데이터가 없습니다.")
        return
        
    print(f"\n🚀 데이터 확인: Train {train_count:,}장 / Val {val_count:,}장")
    print("🤖 제품 불량 판독 AI 엔진 학습을 준비합니다...")
    
    # [수정] 안내서(.yaml) 파일 역시 완전히 새로운 폴더 안에 저장되도록 변경했습니다.
    yaml_path = os.path.abspath(os.path.join(BASE_DATA_DIR, "product_inspection.yaml"))
    data_config = {
        "path": os.path.abspath(BASE_DATA_DIR),  # AI에게 새로운 폴더의 절대 경로를 알려줍니다.
        "train": "images/train",  
        "val": "images/val",    
        "nc": len(CLASS_NAMES),   # 2개
        "names": CLASS_NAMES      # ["Product_Normal", "Product_Break"]
    }
    
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, allow_unicode=True, sort_keys=False)
    
    # 모델 불러오기 (v8n 혹은 사용자 정의 Nano 모델)
    try:
        model = YOLO("yolov8n.pt") # 범용성이 높은 v8 nano 모델을 기본 권장
    except Exception as e:
        print("⚠️ 모델 다운로드 오류가 발생했습니다. 인터넷 연결을 확인하세요.")
        return

    print("\n[INFO] AI 학습 엔진 가동...")
    
    # [변경 4] 학습 모델 이름 변경
    model_name = "product_inspection_nano"
    model.train(
        data=yaml_path,     
        epochs=50,          
        imgsz=320,          
        batch=16,           
        cache=True,         
        workers=0,          
        amp=True,           
        patience=10,        
        project="runs/train", 
        name=model_name     
    )
    
    final_weights_path = os.path.abspath(f"runs/train/{model_name}/weights/best.pt")
    print("\n🎉 [축하합니다!] 제품 양불 판독 AI 모델 학습이 완료되었습니다!")
    print(f"👉 완성된 모델 파일: '{final_weights_path}'")

    # DB에 학습 기록 저장
    try:
        conn = pymysql.connect(host=DB_CONFIG['host'], user=DB_CONFIG['user'], password=DB_CONFIG['password'], database=DB_NAME, charset=DB_CONFIG['charset'])
        cursor = conn.cursor()
        sql = "INSERT INTO training_logs (model_name, weights_path, trained_at) VALUES (%s, %s, %s)"
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute(sql, (model_name, final_weights_path, now_str))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB ERROR] 학습 이력 데이터베이스 저장 실패: {e}")

# ==========================================
# [조종실] 실제 코드가 실행되는 흐름
# ==========================================
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    print("=" * 60)
    init_db() # 제품 검사용 새 DB(product_inspection_db) 생성

    print("\n" + "=" * 60)
    ans = input("❓ [1단계] 두 영상(정상/파손)에서 사진을 추출하시겠습니까? (y/n): ")
    if ans.lower() == 'y':  
        # 딱 2개의 제품 영상만 처리하도록 리스트 축소
        # 업로드해주신 파일명과 정확히 일치하도록 대소문자를 맞췄습니다.
        video_datasets = [
            {"video_path": "Product_Normal.mp4", "class_id": 0, "class_name": "Product_Normal"},
            {"video_path": "product_Break.mp4", "class_id": 1, "class_name": "Product_Break"}
        ]
        
        for data in video_datasets:
            # 짧은 영상에서 사진을 아주 촘촘하게 추출하도록 step=3 유지
            extract_and_preprocess(data["video_path"], data["class_id"], data["class_name"], step=3)
            
    print("\n" + "=" * 60)
    # [수정] 안내 문구에도 새로운 폴더 경로(BASE_DATA_DIR)가 제대로 출력되도록 반영했습니다.
    print(f"⚠️ 필수 안내: '{BASE_DATA_DIR}/images/representative_samples' 폴더에 저장된 30장의 이미지에")
    print("              라벨링 툴(LabelImg 등)을 사용하여 정답 박스(.txt)를 꼭 그려주세요.")
    print("              특히 '파손' 부위나 제품 전체의 위치를 AI에게 알려주는 아주 중요한 작업입니다!")
    print("=" * 60)
    
    ans = input("❓ [2단계] 사용자 라벨링이 완료되었습니까? 지금 바로 AI 학습을 시작할까요? (y/n): ")
    if ans.lower() == 'y':
        train_yolo_model()
    else:
        print("작업을 임시 종료합니다. 라벨링 완료 후 이 코드를 다시 실행하여 '2단계'만 진행해 주세요.")
import cv2          # [영상 처리] 동영상을 읽고 이미지를 자르거나 크기를 줄이는 데 사용하는 라이브러리입니다.
import os           # [파일/폴더 관리] 컴퓨터의 폴더를 만들거나 파일이 있는지 확인하는 역할을 합니다.
import yaml         # [설정 파일 생성] 인공지능이 학습할 때 참고할 '데이터 지도(.yaml)'를 만드는 데 쓰입니다.
import random       # [랜덤 뽑기] 수많은 사진 중 무작위로 Train(학습용)과 Val(검증용/모의고사)을 나누기 위해 사용합니다.
import numpy as np  # [수학 계산] 이미지 데이터를 컴퓨터가 이해할 수 있는 숫자로 변환할 때 보조해줍니다.
import pymysql      # [데이터베이스] 추출된 데이터와 학습 기록을 MySQL DB에 꼼꼼히 적어두는 '기록장' 역할을 합니다.
from datetime import datetime  # [시간 기록] 파일 이름에 현재 시간을 넣거나 DB에 저장 시간을 기록할 때 씁니다.
from ultralytics import YOLO   # [AI 핵심] 드론 부품을 인식하게 될 YOLO 인공지능 모델을 불러오는 핵심 라이브러리입니다.

# ==========================================
# [설정 1] 인공지능이 외워야 할 '정답지 목록' (사용자 정의 학습 분야)
# ==========================================
# 여기에 적힌 이름의 순서(0번부터 시작)대로 인공지능이 부품을 인식하게 됩니다.
CLASS_NAMES = [
    "UAV_Wheel", "Left_wing", "Right_wing", "Fuselage_Part_1", "Fuselage_Part_2_1", 
    "Fuselage_Part_2_2", "Fuselage_Part_3_1", "Fuselage_Part_3_2", "Fuselage_Part_4_1", 
    "ISR_System_1", "ISR_System_2", "Jet_Engine_1", "Jet_Engine_2", 
    "Missile_Launcher", "Missile", "Pylon", "UAV_Fuselage_1"
]

# ==========================================
# [설정 2] DB 연결 정보 (작업 내역을 저장할 MySQL 금고)
# ==========================================
DB_CONFIG = {
    'host': 'localhost',      # DB가 설치된 컴퓨터 주소 (내 컴퓨터면 localhost)
    'user': 'root',           # DB 관리자 아이디
    'password': '1234',       # DB 관리자 비밀번호 (본인 환경에 맞게 꼭 수정해야 함)
    'charset': 'utf8mb4'      # 한글 깨짐을 방지하는 문자 설정
}
DB_NAME = 'drone_vision_db'   # 우리가 사용할 전용 데이터베이스 이름

def init_db():
    """
    [기능 0] 데이터베이스 준비하기
    프로그램이 켜지면 제일 먼저 실행되며, MySQL에 'drone_vision_db'라는 금고가 없으면 새로 만들고,
    사진 목록과 학습 기록을 적을 '표(테이블)' 2개를 자동으로 생성해줍니다.
    """
    try:
        # 1. 데이터베이스(금고) 자체를 생성합니다. (이미 있으면 건너뜀)
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
        
        # [테이블 A] 추출한 사진의 이름을 기록하는 표
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extracted_images (
                id INT AUTO_INCREMENT PRIMARY KEY, -- 번호표 (1번, 2번 순서대로 자동 부여)
                class_name VARCHAR(50),            -- 드론 부품 이름 (예: UAV_Wheel)
                file_name VARCHAR(255),            -- 사진 파일 이름
                dataset_type VARCHAR(10),          -- 학습용(train)인지 모의고사용(val)인지 구분
                extracted_at DATETIME              -- 사진을 뽑아낸 정확한 시간
            )
        """)
        
        # [테이블 B] AI 학습이 끝난 후 모델 파일이 어디에 저장되었는지 기록하는 표
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                model_name VARCHAR(100),           -- 만들어진 AI 모델의 이름
                weights_path VARCHAR(255),         -- AI의 뇌(가중치 파일)가 저장된 컴퓨터 내 위치
                trained_at DATETIME                -- 학습이 완료된 시간
            )
        """)
        conn.commit()
        conn.close()
        print("[DB] MySQL 데이터베이스 초기화 및 연결 성공! (기록할 준비 완료)")
    except Exception as e:
        print(f"[DB ERROR] MySQL 연결 실패. 비밀번호나 서버 상태를 확인하세요: {e}")

# ==========================================
# [기능 1] 동영상에서 사진 뽑아내기 & Train/Val 분리 & 30장 대표 샘플 뽑기
# ==========================================
def extract_and_preprocess(video_path, class_id, class_name, step=15):
    """
    이 함수는 동영상 1개를 받아서 아래 3가지 일을 합니다:
    1. 사진 크기를 320x320으로 팍 줄여서 하드디스크 용량을 아낍니다.
    2. 10장 중 8장은 Train(공부용) 폴더로, 2장은 Val(시험용) 폴더로 나눕니다.
    3. 사람이 라벨링(박스치기)하기 쉽도록 영상 전체에서 골고루 30장만 뽑아 '대표 샘플 폴더'에 따로 저장합니다.
    """
    
    # 동영상 파일이 실제로 있는지 검사합니다. 없으면 경고문구를 띄우고 취소합니다.
    if not os.path.exists(video_path):
        print(f"  [-] 누락: '{video_path}' 파일을 찾을 수 없습니다. (경로를 확인하세요)")
        return

    # 사진들이 저장될 3개의 방(폴더) 이름을 정합니다.
    train_dir = "datasets/images/train"                       # 80%의 공부용 사진이 들어갈 곳
    val_dir = "datasets/images/val"                           # 20%의 모의고사용 사진이 들어갈 곳
    sample_dir = "datasets/images/representative_samples"     # 라벨링을 위한 30장 요약본이 들어갈 곳
    
    # 위에서 정한 방(폴더)이 없으면 새로 만들어줍니다.
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    # 영상을 재생할 준비를 합니다.
    cap = cv2.VideoCapture(video_path)
    
    # --- [30장 스마트 추출 계산기] ---
    # 영상의 총 길이를 먼저 확인하고, 30장을 골고루 뽑기 위해 몇 장마다 한 번씩 챙겨야 할지 계산합니다.
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 영상 전체 프레임 수
    estimated_total_extracts = total_frames // step        # step(15)마다 뽑았을 때 예상되는 전체 사진 수
    sample_interval = max(1, estimated_total_extracts // 30) # 전체 사진 수에서 30을 나눠서 간격을 구함
    
    # 개수를 세기 위한 카운터(계산기)들 0으로 초기화
    saved_train = 0
    saved_val = 0
    sample_saved = 0
    extract_count = 0 
    
    # DB에 한 번에 쏟아붓기 위해 기록을 담아둘 빈 바구니 생성
    db_records = []

    print(f"\n▶ [{class_name}] 영상 추출 시작 (해상도: 320x320로 압축 중) ...")
    
    while True:
        # 영상에서 딱 한 장면(프레임)을 찰칵 찍어옵니다.
        ret, frame = cap.read()
        if not ret: 
            break # 영상이 끝나서 더 이상 찍을 사진이 없으면 멈춥니다(break).

        # [용량 다이어트] 4K나 1080p 해상도면 용량이 너무 크니, 가로 320 세로 320으로 사진을 꾹꾹 눌러 작게 만듭니다.
        frame_resized = cv2.resize(frame, (320, 320))
        
        # 파일 이름을 겹치지 않게 '부품명_년월일시분초_순서.jpg'로 멋지게 지어줍니다.
        now_time = datetime.now()
        base_filename = f"{class_name}_{now_time.strftime('%Y%m%d%H%M%S')}_{extract_count:04d}.jpg"
        
        # [운명의 주사위 굴리기: Train vs Val]
        # random.random()은 0~1 사이의 숫자를 뽑아줍니다. 0.2보다 작을 확률은 20%입니다.
        is_val = random.random() < 0.2 
        
        # 20% 확률에 당첨되면 val 폴더로, 아니면 train 폴더로 목적지를 정합니다.
        dataset_type = "val" if is_val else "train"
        target_dir = val_dir if is_val else train_dir
        
        # 최종적으로 사진이 저장될 폴더 경로와 파일 이름을 합칩니다.
        target_filepath = os.path.join(target_dir, base_filename)
        
        # 한글 경로 에러를 피하기 위해 cv2.imencode를 사용하여 안전하게 사진을 저장합니다.
        is_success, img_arr = cv2.imencode('.jpg', frame_resized)
        if is_success:
            with open(target_filepath, 'wb') as f:
                img_arr.tofile(f)
            
            # 카운터 1 증가
            if is_val:
                saved_val += 1
            else:
                saved_train += 1

            # DB 바구니에 방금 저장한 사진의 정보(부품명, 파일명, 학습타입, 시간)를 담아둡니다.
            db_records.append((class_name, base_filename, dataset_type, now_time.strftime('%Y-%m-%d %H:%M:%S')))
            
            # --- [사람을 돕는 30장 샘플 저장 기능] ---
            # extract_count가 계산해둔 간격(interval)에 딱 맞아 떨어지고, 아직 30장을 다 못 채웠다면?
            if extract_count % sample_interval == 0 and sample_saved < 30:
                sample_filename = f"SAMPLE_{class_name}_{sample_saved+1:02d}.jpg"
                sample_filepath = os.path.join(sample_dir, sample_filename)
                
                # 방금 그 장면을 '대표 샘플 폴더'에도 복사본으로 몰래 넣어둡니다.
                with open(sample_filepath, 'wb') as f:
                    img_arr.tofile(f)
                sample_saved += 1 # 샘플 수집 카운터 1 증가
        
        extract_count += 1
        # 화면에 현재 진행 상황을 덮어쓰기(\r)로 깔끔하게 보여줍니다.
        print(f"  - 진행: 공부용 {saved_train}장 / 모의고사용 {saved_val}장 (대표 샘플 {sample_saved}/30 분리 중...)", end="\r")

        # 너무 많은 사진이 찍히는 걸 막기 위해, 설정한 step(기본값 15장) 만큼 영상을 빨리감기 하듯 건너뜁니다.
        for _ in range(step - 1):
            if not cap.grab():
                break

    # 영상 재생기 전원 끄기
    cap.release()
    print(f"\n  ✅ [{class_name}] 총 {saved_train + saved_val}장 (Train:{saved_train}, Val:{saved_val}) 추출 및 분류 완료!")

    # [DB 일괄 저장] 바구니에 모인 수천 개의 기록을 MySQL 금고에 1초 만에 쏟아부어서 저장합니다.
    if db_records:
        try:
            conn = pymysql.connect(host=DB_CONFIG['host'], user=DB_CONFIG['user'], password=DB_CONFIG['password'], database=DB_NAME, charset=DB_CONFIG['charset'])
            cursor = conn.cursor()
            sql = "INSERT INTO extracted_images (class_name, file_name, dataset_type, extracted_at) VALUES (%s, %s, %s, %s)"
            cursor.executemany(sql, db_records) # executemany: 리스트에 있는 데이터를 한 방에 저장하는 마법의 명령어
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"  [DB ERROR] 데이터베이스 저장 실패: {e}")

# ==========================================
# [기능 2] YOLO26-Nano 커스텀 모델 학습 (인공지능 훈련소)
# ==========================================
def train_yolo_model():
    """
    이 함수는 사용자가 직접 라벨링(정답지 만들기)을 마친 데이터들을 바탕으로
    초경량·초고속 AI인 YOLO26-Nano 모델을 교육시켜 나만의 드론 부품 인식 AI를 만드는 역할을 합니다.
    """
    train_dir = "datasets/images/train"
    val_dir = "datasets/images/val"
    
    # 폴더가 없으면 AI가 공부할 책이 없는 것이므로 경고를 띄웁니다.
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("🔴 학습 또는 검증 데이터 폴더가 없습니다. [1단계] 영상을 먼저 추출해주세요.")
        return
        
    # 폴더 안에 사진이 몇 장이나 있는지 셉니다.
    train_count = len([f for f in os.listdir(train_dir) if f.endswith('.jpg')])
    val_count = len([f for f in os.listdir(val_dir) if f.endswith('.jpg')])
    
    if train_count == 0 or val_count == 0:
        print("🔴 폴더 내에 학습(Train) 또는 검증(Val) 데이터가 부족합니다.")
        return
        
    print(f"\n🚀 데이터 확인: Train {train_count:,}장 / Val {val_count:,}장")
    print("🤖 최신 YOLO26-Nano 엔진으로 학습을 준비합니다...")
    
    # [데이터 지도 만들기]
    # AI에게 "사진 폴더는 여기 있고, 네가 맞혀야 할 부품 이름은 이거야"라고 알려주는 .yaml 안내서를 만듭니다.
    yaml_path = os.path.abspath("datasets/drone_parts.yaml")
    data_config = {
        "path": os.path.abspath("datasets"),
        "train": "images/train",  
        "val": "images/val",    
        "nc": len(CLASS_NAMES),   # 총 부품 개수 (nc: number of classes)
        "names": CLASS_NAMES      # 부품 이름 목록
    }
    
    # 안내서(.yaml)를 파일로 저장합니다.
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, allow_unicode=True, sort_keys=False)
    
    # [AI 두뇌 불러오기]
    # YOLO26-Nano라는 텅 빈 똑똑한 뇌(초기 모델)를 가져옵니다.
    try:
        model = YOLO("yolov26n.pt") 
    except Exception as e:
        print("⚠️ 주의: yolov26n.pt 파일을 찾을 수 없거나 다운로드할 수 없습니다.")
        print("만약 공식 지원 파일이 아니라면, 현재 사용 가능한 최신 Nano 모델(예: yolov8n.pt)을 사용해야 합니다.")
        return

    print("\n[INFO] AI 학습 엔진 가동 (입력 사진 크기는 320에 최적화되었습니다)...")
    
    # [본격적인 스파르타 교육 시작!]
    model_name = "drone_yolo26_nano"
    model.train(
        data=yaml_path,     # 아까 만든 데이터 지도(.yaml)를 줍니다.
        epochs=50,          # 책(데이터)을 처음부터 끝까지 총 50번 반복해서 읽도록 합니다.
        imgsz=320,          # 사진 크기가 320x320이라는 것을 AI에게 알려줍니다.
        batch=16,           # 한 번에 16장의 사진을 동시에 외우게 하여 학습 속도를 높입니다.
        cache=True,         # 사진을 컴퓨터 램(RAM)에 올려두어 빛의 속도로 읽어오게 합니다.
        workers=0,          # Windows 에러 방지를 위해 데이터 운반자를 0으로 설정합니다.
        amp=True,           # '자동 혼합 정밀도'라는 기술을 켜서 그래픽카드의 학습 속도를 비약적으로 올립니다.
        patience=10,        # 만약 10번이나 반복해서 읽었는데도 실력이 더 이상 안 오르면, 50번을 덜 채워도 조기 졸업 시킵니다.
        project="runs/train", # 학습 결과물(성적표, 완성된 뇌)을 저장할 최상위 폴더 이름입니다.
        name=model_name     # 이번 학습의 프로젝트 이름입니다.
    )
    
    # AI의 훈련이 끝나고 가장 똑똑했던 순간의 뇌(best.pt)가 저장된 위치를 찾습니다.
    final_weights_path = os.path.abspath(f"runs/train/{model_name}/weights/best.pt")
    print("\n🎉 [축하합니다!] YOLO26-Nano 커스텀 모델 학습이 성공적으로 완료되었습니다!")
    print(f"👉 완성된 모델(뇌) 파일 위치: '{final_weights_path}'")

    # DB 테이블에 '이 모델은 언제 만들어졌고 파일은 어디에 있다'라고 최종 이력을 남겨둡니다.
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
# [조종실] 코드를 실행하면 여기서부터 순서대로 작동합니다!
# ==========================================
if __name__ == "__main__":
    # 윈도우(Windows) 컴퓨터에서 코드가 2~3개씩 꼬여서 동시에 켜지는 버그를 막아주는 필수 마법의 주문입니다.
    import multiprocessing
    multiprocessing.freeze_support()
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    print("=" * 60)
    init_db() # 프로그램 시작 전 DB를 세팅합니다.

    print("\n" + "=" * 60)
    # 1단계 질문
    ans = input("❓ [1단계] 영상을 추출하여 Train/Val 분리 및 대표 샘플 30장 추출을 진행할까요? (y/n): ")
    if ans.lower() == 'y':  
        # 17개의 영상 파일 이름과 부품 이름을 짝지어둔 리스트입니다.
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
        
        # 반복문을 돌면서 17개의 영상을 하나하나 추출 함수(extract_and_preprocess)로 보냅니다.
        for data in video_datasets:
            extract_and_preprocess(data["video_path"], data["class_id"], data["class_name"])
            
    print("\n" + "=" * 60)
    print("⚠️ 필수 안내: 'datasets/images/representative_samples' 폴더에 저장된 30장의 이미지에")
    print("              라벨링 툴(LabelImg 등)을 사용하여 정답 박스(.txt)를 그리는 작업을 꼭 해주세요.")
    print("              라벨링이 끝난 후에는 그 이미지와 txt 파일들을 Train/Val 폴더로 복사해 주셔야 합니다!")
    print("=" * 60)
    
    # 2단계 질문
    ans = input("❓ [2단계] 사용자 라벨링이 완료되었습니까? 지금 바로 YOLO26-Nano AI 학습을 시작할까요? (y/n): ")
    if ans.lower() == 'y':
        train_yolo_model() # 학습 함수 실행
    else:
        print("작업을 종료합니다. 라벨링 완료 후 다시 코드를 실행하여 2단계만 진행해 주세요.")
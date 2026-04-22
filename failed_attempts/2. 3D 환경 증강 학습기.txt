import os           # 운영체제(Windows)와 소통하여 폴더를 만들고, 파일 존재 여부를 확인하는 라이브러리입니다.
import shutil       # 파일이나 폴더를 다른 곳으로 '이동(Move)'시키거나 복사할 때 사용하는 강력한 파일 관리 라이브러리입니다.
import yaml         # YOLO 모델이 읽을 수 있는 데이터 안내서(data.yaml) 파일을 생성하고 작성하기 위한 라이브러리입니다.
from ultralytics import YOLO  # 딥러닝 객체 탐지 인공지능인 YOLOv8(또는 26)을 불러오고 제어하는 핵심 라이브러리입니다.

def setup_dataset_and_train():
    """
    [3D 다각도 인식 실패 예방 2단계]
    이 함수는 크게 두 가지 역할을 합니다.
    1. 사람이 막 라벨링을 끝낸 원본 폴더를 AI(YOLO)가 인식할 수 있는 전용 폴더 구조로 '자동 청소/정리' 합니다.
    2. 데이터가 부족하더라도 3D 환경을 시뮬레이션하는 '데이터 증강' 기법을 써서 AI를 훈련시킵니다.
    """
    
    # ---------------------------------------------------------
    # 1. 데이터셋 폴더 경로 설정
    # ---------------------------------------------------------
    # base_dir은 방금 AnyLabeling 작업을 마친 로컬 PC의 절대 경로(주소)입니다.
    # 앞에 'r'이 붙은 이유는 Windows의 역슬래시(\) 문법 오류를 무시하고 텍스트 그대로 읽게 하기 위함입니다.
    base_dir = r"C:\Users\Ghost\OneDrive\New Robotics Study File\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\dataset"
    
    # 원본 이미지가 들어있는 폴더와, AnyLabeling이 만들어낸 txt 파일(라벨)이 들어있는 폴더의 주소입니다.
    raw_images_dir = os.path.join(base_dir, "raw_images")
    labels_dir = os.path.join(base_dir, "labels")
    
    # YOLO는 까다롭게도 반드시 'images/train', 'labels/train' 이라는 이름의 폴더 구조를 요구합니다.
    # 그 요구사항에 맞출 목표 폴더 주소를 미리 만들어 둡니다.
    images_train_dir = os.path.join(base_dir, "images", "train")
    labels_train_dir = os.path.join(base_dir, "labels", "train")
    
    # ---------------------------------------------------------
    # 2. YOLO 인식 오류를 막기 위한 폴더 구조 자동 재배치
    # ---------------------------------------------------------
    print("📂 데이터셋 폴더 구조를 YOLO 표준 형식에 맞게 자동 정리합니다...")
    
    # [이미지 이동 작업]
    if os.path.exists(raw_images_dir): # 만약 raw_images 폴더가 존재한다면
        os.makedirs(images_train_dir, exist_ok=True) # 목표 폴더(images/train)를 만듭니다. (이미 있으면 무시)
        
        # raw_images 폴더 안에 있는 모든 파일을 하나씩 꺼내서 확인합니다.
        for f in os.listdir(raw_images_dir):
            file_path = os.path.join(raw_images_dir, f)
            if os.path.isfile(file_path): # 그것이 폴더가 아니라 파일(이미지)이라면
                shutil.move(file_path, os.path.join(images_train_dir, f)) # images/train 폴더로 이동시킵니다.
        
        # 이동이 끝나서 raw_images 폴더가 텅 비었다면, 헷갈리지 않게 껍데기 폴더를 지워버립니다.
        if not os.listdir(raw_images_dir):
            try:
                os.rmdir(raw_images_dir) # 폴더 삭제 시도
            except PermissionError:
                # 단, 폴더를 윈도우 탐색기나 AnyLabeling 프로그램이 꽉 잡고 있으면 삭제 권한 에러가 납니다.
                # 이때 프로그램이 뻗지(Crash) 않도록 에러를 무시하고 넘어가게 하는 안전장치입니다.
                print(f"⚠️ 알림: '{raw_images_dir}' 폴더가 다른 프로그램에서 사용 중이라 삭제하지 못했습니다. (학습에는 영향이 없습니다.)")
            except Exception as e:
                pass
            
    # [라벨(txt) 이동 작업]
    if os.path.exists(labels_dir): # 만약 labels 폴더가 존재한다면
        # labels 폴더 안에 있는 여러 파일 중, 폴더를 제외한 진짜 파일들(txt 등)만 골라내어 리스트로 만듭니다.
        txt_files = [f for f in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, f))]
        
        if txt_files: # 파일이 하나라도 있다면
            os.makedirs(labels_train_dir, exist_ok=True) # 목표 폴더(labels/train)를 생성합니다.
            for f in txt_files:
                shutil.move(os.path.join(labels_dir, f), os.path.join(labels_train_dir, f)) # 라벨 파일들을 목표 폴더로 이동시킵니다.
                
        # 라벨 파일 이동 후, labels 폴더 바로 아래가 텅 비었다면 껍데기 폴더를 삭제합니다.
        try:
            if not os.listdir(labels_dir):
                os.rmdir(labels_dir)
        except PermissionError:
            # 여기도 마찬가지로 파일 탐색기 등이 폴더를 열고 있을 때 발생하는 에러를 부드럽게 넘깁니다.
            print(f"⚠️ 알림: '{labels_dir}' 폴더가 다른 프로그램에서 사용 중이라 삭제하지 못했습니다. (학습에는 영향이 없습니다.)")
        except Exception as e:
            pass

    # ---------------------------------------------------------
    # 3. classes.txt를 바탕으로 data.yaml 자동 생성
    # ---------------------------------------------------------
    # YOLO에게 "네가 맞춰야 할 정답(클래스)의 이름과 번호는 이거야!"라고 알려주는 설정 파일을 만듭니다.
    # 이 순서는 AnyLabeling에서 작업할 때 생긴 classes.txt의 순서와 정확히 일치해야 합니다. (0: Fuselage_3_1, 1: ISR_System, 2: Right_wing)
    classes = ["Fuselage_3_1", "ISR_System", "Right_wing"]
    yaml_path = os.path.join(base_dir, "data.yaml") # 데이터셋 폴더 최상단에 data.yaml 파일을 만들 예정입니다.
    
    # yaml 파일에 들어갈 내용(딕셔너리 구조)입니다.
    yaml_data = {
        "path": base_dir,       # 전체 데이터셋이 있는 최상위 폴더 경로
        "train": "images/train",# AI가 공부할 교과서(학습 이미지)가 있는 폴더 위치 (path 기준 상대경로)
        "val": "images/train",  # AI가 모의고사를 칠(검증) 이미지가 있는 곳. 현재 데이터가 적어 교과서로 모의고사도 칩니다.
        "nc": len(classes),     # 넘버 오브 클래스(Number of Classes): 찾을 부품의 총 개수 (여기선 3개)
        "names": classes        # 각 번호(0, 1, 2)에 해당하는 실제 부품의 이름표 리스트
    }
    
    # 작성한 내용을 실제 파일(data.yaml)로 하드디스크에 저장합니다.
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False)
        
    print(f"✅ 설정 파일 생성 완료: {yaml_path}")
    
    # ---------------------------------------------------------
    # 4. 본격적인 AI 모델 학습 시작 (Data Augmentation)
    # ---------------------------------------------------------
    # 가벼우면서도 성능이 뛰어나 실시간 카메라 환경에 적합한 YOLOv8 'Nano(n)' 모델의 기본 뼈대를 불러옵니다.
    model = YOLO("yolov8n.pt") 
    print("\n🚀 3D 다각도 인식을 위한 증강(Augmentation) 학습을 시작합니다...")

    # .train() 함수를 실행하여 본격적으로 공부를 시작합니다. AI 모델이 사진을 보고 박스를 치는 연습을 합니다.
    results = model.train(
        data=yaml_path,             # 방금 위에서 만든 안내서(data.yaml)를 보고 데이터를 찾아갑니다.
        epochs=150,                 # 150번 반복해서 데이터셋을 달달 외울 때까지 공부합니다. (너무 적으면 못 맞추고, 너무 많으면 과적합 발생)
        imgsz=640,                  # 입력되는 사진을 정사각형(640x640) 크기로 강제로 맞추어 공부합니다. (YOLO의 표준 해상도)
        batch=16,                   # 한 번에 16장의 사진을 그래픽카드(메모리)에 올려서 공부합니다. (GPU 메모리가 부족해 에러가 나면 8이나 4로 줄이세요)
        project="Drone_QC_Project", # 학습 결과물(그래프, 모델 파일 등)이 저장될 최상위 폴더 이름입니다.
        name="robust_3d_model",     # 그 안에서 현재 진행하는 학습의 프로젝트 이름입니다.
        
        # =======================================================
        # ★ [핵심] 3D 다각도 인식 실패를 예방하는 마법의 데이터 증강(Augmentation) 파라미터들 ★
        # 이 설정값들이 원본 사진을 이리저리 비틀고 조작해서, AI가 3D 공간의 각도 변화를 이해하게 만듭니다.
        # =======================================================
        degrees=45.0,       # [회전] 부품 사진을 시계/반시계 방향으로 무작위로 최대 45도까지 돌려버립니다. 컨베이어에서 부품이 삐딱하게 놓일 때를 대비합니다.
        translate=0.2,      # [이동] 부품을 사진 중심에서 상하좌우로 20% 이내로 밀어버립니다. 부품이 카메라 정중앙이 아닌 가장자리에 찍힐 때를 대비합니다.
        scale=0.5,          # [크기 축소/확대] 사진 크기를 50%까지 줄이거나 키웁니다. 드론 부품이 카메라에서 멀어지거나 가까워질 때의 3D 원근감을 학습합니다.
        shear=15.0,         # [전단/기울기] 사진의 모서리를 잡아당겨 평행사변형처럼 15도까지 찌그러뜨립니다. 3D 입체물이 비스듬하게 찍혔을 때의 왜곡을 흉내 냅니다.
        perspective=0.001,  # [원근법] (매우 중요) 카메라 렌즈의 각도가 상하좌우로 틀어지는 완벽한 3D 효과(투시)를 강제로 부여합니다.
        flipud=0.5,         # [상하 반전] 50% 확률로 사진을 위아래로 뒤집습니다. 부품이 거꾸로 조립라인에 들어올 경우를 대비합니다.
        fliplr=0.5,         # [좌우 반전] 50% 확률로 사진을 좌우로 거울처럼 뒤집습니다. 대칭 부품 인식에 아주 좋습니다.
        mosaic=1.0,         # [모자이크] 무조건(100%) 4장의 사진을 십자가 형태로 섞어 1장으로 만듭니다. 부품의 일부분이 다른 것에 가려져도 인식하게 하는 현존 최고의 방어 기법입니다.
        
        # --- 조명 환경 변화 시뮬레이션 ---
        hsv_h=0.015,        # [색상 변화] 공장 조명의 색온도(누런빛, 푸른빛)가 미세하게 바뀔 때를 대비해 색조(Hue)를 약간 비틉니다.
        hsv_s=0.7,          # [채도 변화] 사진을 흑백에 가깝게 빼거나 아주 쨍하게 만듭니다. 그림자나 강한 조명 반사를 견디게 합니다.
        hsv_v=0.4           # [명도 변화] 공장 내부가 어두워지거나 카메라 플래시가 터져서 너무 밝아질 때를 대비하여 사진의 밝기(Value)를 무작위로 40%까지 바꿉니다.
    )
    
    # 학습이 무사히 끝나면 화면에 안내 문구를 띄웁니다.
    print("\n✅ 학습이 완료되었습니다! best.pt 파일이 Drone_QC_Project/robust_3d_model/weights/ 폴더에 생성되었습니다.")

# 파이썬에서 해당 파일을 직접 실행했을 때만 이 아래 코드가 작동하도록 하는 표준 명령어입니다.
if __name__ == "__main__":
    # 로컬 환경(Windows PC)에서 파이썬 코드가 멀티프로세싱(여러 두뇌를 동시에 씀)을 할 때 무한 루프에 빠지는 치명적인 에러를 막기 위해
    # 반드시 함수(setup_dataset_and_train)를 껍데기(if __name__) 안에서 호출해야 합니다.
    setup_dataset_and_train()
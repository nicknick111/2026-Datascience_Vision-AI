# vision_ai_main.py
import cv2
import math
import torch
import os
import time
import threading
import queue
import datetime
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from PIL import Image

# ==========================================
# [0. 공유 상태 변수 (웹/API 서버와의 통신용)]
# ==========================================
lock = threading.Lock()           # [V1 동일] API 서버가 화면을 가져갈 때 화면이 깨지지 않도록 보호하는 자물쇠
global_frame = None               # 웹 모니터링으로 송출될 프레임 복사본
trigger_flag = False              # PLC에서 검사 명령이 떨어지면 True가 됨
is_running = True                 # 메인 루프 동작 상태

current_status = "WAITING"        # [V1 대비 개선] 단순 결과(PASS/FAIL)뿐 아니라 현재 상태(대기, 검사중)도 관리
current_confidence = 0.0          # API로 전달할 최종 AI 확신도
current_part_name = ""            # API로 전달할 판정된 부품명
current_order_id = ""             # 통신 추적을 위한 고유 ID
expected_product = "Normal_C4" 

log_queue = queue.Queue(maxsize=100) # 터미널 로그를 API 서버로 넘기기 위한 바구니

# ==========================================
# [1. 비동기 데이터 수집 (백그라운드 저장)] - ✨ V1에 없던 핵심 신기능
# ==========================================
COLLECT_DIR = r"E:\Robot_Team_Project\Collected_Data"
last_collect_time = 0.0
save_queue = queue.Queue()        # 디스크 저장을 대기하는 이미지들이 임시로 머무는 메모리 바구니

def background_image_saver():
    """
    [개선점] V1에서는 메인 루프에서 이미지를 직접 저장(cv2.imwrite)하면,
    디스크 I/O 딜레이 때문에 카메라 영상이 뚝뚝 끊기는 프레임 드랍 현상이 발생했습니다.
    이제는 이 백그라운드 스레드가 메인 루프와 별개로 조용히 이미지를 꺼내서 저장합니다.
    """
    os.makedirs(COLLECT_DIR, exist_ok=True)
    while True:
        filepath, img_data = save_queue.get()
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            cv2.imwrite(filepath, img_data) 
        except Exception as e:
            print(f"저장 실패: {e}")
        save_queue.task_done()

# 프로그램 시작 시 백그라운드 저장 요정(스레드)을 출근시킵니다.
threading.Thread(target=background_image_saver, daemon=True).start()

def system_log(msg):
    """콘솔 출력과 동시에 API 로그 큐에 메시지를 담습니다."""
    timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)
    if not log_queue.full(): log_queue.put(full_msg)

# ==========================================
# [2. 환경 설정 및 AI 전처리]
# ==========================================
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
YOLO_MODEL_PATH = r"E:\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\runs\obb\train\weights\best.pt"
EFFNET_WEIGHTS_PATH = r"E:\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\best_effnet_weights_V2_safe.pth"
CLASS_NAMES = ['Defect', 'Normal_C4', 'Normal_Comm', 'Normal_Supply']

# 렌즈 왜곡 보정용 매트릭스 (어안 렌즈처럼 휘어지는 현상 방지)
MTX = np.array([[922.75191915, 0.0, 349.17195886], [0.0, 921.84949514, 258.9181156], [0.0, 0.0, 1.0]])
DIST = np.array([[-0.05206979, -0.45033785, 0.00167389, -0.00407981, 1.2589255]])

# ✨ [V1에 없던 기능] AI 2단 정확도 필터
MIN_PASS_ACCURACY = 80.0   # 이 수치 이상이어야만 정상 판독으로 인정
IGNORE_ACCURACY = 50.0     # 50% 미만은 빛 반사 노이즈로 간주하고 무시

def apply_advanced_preprocessing(part_img):
    """
    [개선점] V1의 단순 필터를 넘어, 가중치 합을 1.0(1.5-0.5)으로 맞춰
    화면이 어두워지는 암전 현상을 해결한 '고급 전처리' 함수입니다.
    """
    filtered = cv2.bilateralFilter(part_img, d=9, sigmaColor=50, sigmaSpace=75)
    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    gaussian = cv2.GaussianBlur(img_clahe, (0, 0), 2.0)
    unsharp = cv2.addWeighted(img_clahe, 1.5, gaussian, -0.5, 0)
    return unsharp

def letterbox_image(img, target_size=224):
    """
    ✨ [개선점] V1에서는 직사각형 부품을 224x224 정사각형으로 강제로 찌그러뜨려 학습을 방해했습니다.
    이 함수는 원본 비율을 유지한 채 남는 공간을 검은색(여백)으로 채워 AI의 인지력을 극대화합니다.
    """
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def load_models():
    detector = YOLO(YOLO_MODEL_PATH)
    classifier = models.efficientnet_b0(weights=None)
    classifier.classifier[1] = nn.Linear(classifier.classifier[1].in_features, len(CLASS_NAMES))
    classifier.load_state_dict(torch.load(EFFNET_WEIGHTS_PATH, map_location=DEVICE))
    classifier = classifier.to(DEVICE).eval()
    return detector, classifier

# [개선점] 레터박스가 이미 크기를 224x224로 맞췄으므로 V1에 있던 Resize()는 제거했습니다.
classifier_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# [3. 메인 추론 파이프라인]
# ==========================================
def run_pipeline(cam_index=0, on_decision_callback=None):
    global global_frame, trigger_flag, is_running, expected_product
    global current_status, current_confidence, current_part_name, current_order_id
    global last_collect_time 
    
    detector, classifier = load_models()
    
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW) 
    cap.set(cv2.CAP_PROP_SETTINGS, 1)        # 실행 시 카메라 하드웨어 설정창 팝업
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 렌즈 왜곡 보정 맵 초기화
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(MTX, DIST, (1280, 720), 1, (1280, 720))
    map1, map2 = cv2.initUndistortRectifyMap(MTX, DIST, None, new_mtx, (1280, 720), cv2.CV_16SC2)

    system_log("비전 엔진이 준비되었습니다.")

    while cap.isOpened() and is_running:
        ret, frame = cap.read()
        if not ret: break

        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        display_img = cv2.resize(undistorted, (960, 540)) # UI 출력용으로 가볍게 축소
        scale_x, scale_y = 960 / 1280, 540 / 720

        # Stage 1: OBB 위치 탐지
        results = detector(undistorted, conf=0.6, verbose=False)
        process_start = time.time() 
        
        # 트리거가 켜지면 상태를 '검사중'으로 변경
        if trigger_flag and current_status != "INSPECTING":
            current_status = "INSPECTING"

        for r in results:
            if r.obb is not None:
                for box in r.obb:
                    cx, cy, bw, bh, angle_rad = box.xywhr[0].cpu().numpy()
                    
                    # 기울어진 부품을 0도로 수평 정렬
                    M = cv2.getRotationMatrix2D((cx, cy), math.degrees(angle_rad), 1.0)
                    rotated = cv2.warpAffine(undistorted, M, (1280, 720))
                    
                    x1, y1 = max(0, int(cx-bw/2)), max(0, int(cy-bh/2))
                    x2, y2 = min(1280, int(cx+bw/2)), min(720, int(cy+bh/2))
                    cropped = rotated[y1:y2, x1:x2]

                    if cropped.size > 0:
                        # [개선점] V1에 없던 고도화된 전처리 + 레터박싱 파이프라인
                        enhanced = apply_advanced_preprocessing(cropped)
                        letterboxed_part = letterbox_image(enhanced, target_size=224)
                        
                        # Stage 2: EfficientNet 판독
                        part_rgb = cv2.cvtColor(letterboxed_part, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(part_rgb)
                        input_tensor = classifier_transform(pil_img).unsqueeze(0).to(DEVICE)
                        
                        with torch.no_grad():
                            output = classifier(input_tensor)
                            probs = torch.softmax(output, 1)
                            conf_score, pred_idx = torch.max(probs, 1)
                        
                        conf_percent = conf_score.item() * 100
                        pure_class_name = CLASS_NAMES[pred_idx.item()]

                        # =========================================================
                        # 💡 [V1 대비 혁신] 정확도 2단 필터 및 폐루프(Active Learning) 수집
                        # =========================================================
                        current_time = time.time()
                        
                        # [필터 1단계] 완전 무시 (허상, 50% 미만)
                        if conf_percent < IGNORE_ACCURACY:
                            continue  

                        # [필터 2단계] 헷갈리는 데이터 자동 수집 (50~80%)
                        elif conf_percent < MIN_PASS_ACCURACY:
                            display_class = f"Check!({pure_class_name})"
                            color = (128, 128, 128) 
                            pred_class = display_class 
                            
                            # 1초 스로틀링: 같은 부품이 계속 찍히며 하드용량이 폭주하는 것을 방지
                            if current_time - last_collect_time > 1.0:
                                filename = f"missed_{int(current_time*1000)}.jpg"
                                filepath = os.path.join(COLLECT_DIR, pure_class_name, filename)
                                # 💡 비동기 큐에 던지기만 하므로 AI 추론 속도에는 전혀 딜레이 없음!
                                save_queue.put((filepath, letterboxed_part.copy()))
                                last_collect_time = current_time
                                print(f"📸 Uncertainty Data Queued: {pure_class_name} ({conf_percent:.1f}%)")
                        
                        # [필터 3단계] 최종 확정 (80% 이상)
                        else:
                            pred_class = pure_class_name
                            color = (0, 0, 255) if pred_class == 'Defect' else (0, 255, 0)

                        # ====================================================
                        # AI 최종 판정 (PLC 통신용 Trigger 로직)
                        # ====================================================
                        if trigger_flag and conf_percent >= MIN_PASS_ACCURACY:
                            trigger_flag = False # 중복 판정 방지
                            current_confidence = conf_percent
                            current_part_name = pred_class
                            
                            if pred_class == 'Defect':
                                current_status = "FAIL"
                            else:
                                current_status = "PASS"
                            
                            elapsed = (time.time() - process_start) * 1000
                            system_log(f"AI 판정 완료: {current_status} ({pred_class}) - {elapsed:.2f}ms")
                            
                            # 외부 서버(api_server)로 결과 전달
                            if on_decision_callback:
                                on_decision_callback(current_status, pred_class)

                        # UI 렌더링 (화면에 OBB 박스와 텍스트 표시)
                        pts = (box.xyxyxyxy[0].cpu().numpy() * [scale_x, scale_y]).astype(int)
                        cv2.polylines(display_img, [pts], True, color, 2)
                        cv2.putText(display_img, f"{pred_class} {conf_percent:.1f}%", (pts[0][0], pts[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 화면 좌측 상단 상태 오버레이
        status_color = (0, 255, 0) if current_status == "PASS" else (0, 0, 255) if current_status == "FAIL" else (0, 255, 255)
        
         # 💡 [UI 수정] 고정된 expected_product 대신, 트리거로 확정된 current_part_name을 출력합니다.
        display_target = current_part_name if current_part_name else "Waiting for Trigger..."
        cv2.putText(display_img, f"Detected: {display_target}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.putText(display_img, f"Status: {current_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # API 서버 스트리밍용으로 복사본 갱신
        with lock:
            global_frame = display_img.copy()

        cv2.imshow("Vision AI", display_img)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            is_running = False
            break

    cap.release()
    cv2.destroyAllWindows()
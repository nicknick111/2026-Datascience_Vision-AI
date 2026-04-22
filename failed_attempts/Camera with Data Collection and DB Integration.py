import os
import cv2
import time
import datetime
import numpy as np
import psycopg2
import requests

# YOLOv8 라이브러리 로드
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] 'ultralytics' 라이브러리가 없습니다. 터미널에 'pip install ultralytics'를 입력하세요.")

# 시스템 설정 (OpenCV 에러 메시지 억제)
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

# ---------------------------------------------------------
# [환경 설정] DB 및 API 
# ---------------------------------------------------------
DB_CONFIG = {
    "host": "localhost",
    "database": "postgres",
    "user": "postgres",
    "password": "1234", # 사용자님의 DB 비밀번호
    "port": "5432"
}
API_URL = "http://10.230.208.237:8000/save"
SAVE_FOLDER = "captured_data"

# [AI 라벨 설정] 학습시킨 3가지 부품의 이름
CUSTOM_LABELS = {
    0: "Cylinder_Part", 
    1: "Blue_Cube",
    2: "Red_Hexahedron"
}

# AI 모델 로드 (학습된 모델이 없으면 기본 모델 사용)
MODEL_PATH = "best.pt" 
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "yolov8n.pt"

try:
    model = YOLO(MODEL_PATH)
    ORIGINAL_NAMES = model.names 
    print(f"[SYSTEM] AI 모델 로드 완료: {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] 모델 로드 실패: {e}")
    model = None

# 이미지 저장 폴더 생성
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

# 전역 상태 변수
is_paused = False
is_running = True
capture_requested = False
capture_count = 0
combined_view = None 
last_log_msg = "READY"
log_expiry_time = 0

# --- [DB 초기 설정] ---
def setup_database():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS image_logs (
                id SERIAL PRIMARY KEY,
                file_path TEXT NOT NULL,
                brightness INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.close()
        conn.close()
        print("[SYSTEM] PostgreSQL 데이터베이스 연결 성공!")
        return True
    except Exception as e:
        print(f"[WARN] DB 연결 실패 (DB가 꺼져있거나 설정이 다릅니다): {e}")
        return False

# --- [조도 자동 보정 로직] ---
def auto_adjust_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    if avg_brightness == 0: avg_brightness = 1
    
    gamma = np.clip(120.0 / avg_brightness, 0.4, 2.5)
    table = np.array([((i / 255.0) ** (1.0/gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    frame_auto = cv2.LUT(frame, table)
    frame_final = np.where(frame_auto > 220, frame_auto * 0.7, frame_auto).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(cv2.cvtColor(frame_final, cv2.COLOR_BGR2GRAY))
    
    return frame_final, gray_clahe, int(avg_brightness)

# --- [마우스 조작 이벤트 (버튼 4개로 확장)] ---
def mouse_callback(event, x, y, flags, param):
    global is_paused, is_running, capture_requested
    if event == cv2.EVENT_LBUTTONDOWN:
        if 470 <= y <= 530:
            if 30 <= x <= 160:    is_paused = False      # PLAY
            elif 180 <= x <= 310: is_paused = True       # PAUSE
            elif 330 <= x <= 460: is_running = False     # EXIT
            elif 480 <= x <= 610: capture_requested = True # CAPTURE

# --- [메인 실행부] ---
def main():
    global is_paused, is_running, capture_requested, capture_count, combined_view
    global last_log_msg, log_expiry_time
    
    setup_database()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Camera not found.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = 'Integrated Vision System'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    prev_time = 0

    while is_running:
        if not is_paused:
            ret, frame = cap.read()
            if not ret: break

            frame_resized = cv2.resize(frame, (960, 540))
            frame_auto, gray_clahe, lux = auto_adjust_lighting(frame_resized)
            
            combined_view = frame_auto.copy()
            detection_info = "None"

            # 1. AI 객체 인식 처리
            if model:
                results = model.predict(source=frame_auto, conf=0.5, verbose=False)
                res = results[0]
                
                for box in res.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    w = x2 - x1
                    h = y2 - y1
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # [지능형 오인식 방지 필터]
                    # - 책상 등 너무 큰 객체는 무시 (화면의 40% 이상 차지 시)
                    if w * h > (960 * 540 * 0.4): continue
                    # - 종횡비가 과도하게 길거나 납작한 경우(책상 모서리 등) 제외
                    if aspect_ratio > 2.0 or aspect_ratio < 0.4: continue

                    # 라벨 매핑 (best.pt 일 때만 커스텀 이름 적용)
                    if MODEL_PATH == "best.pt" and cls_id in CUSTOM_LABELS:
                        label_name = CUSTOM_LABELS[cls_id]
                    else:
                        label_name = ORIGINAL_NAMES[cls_id]
                    
                    detection_info = f"{label_name}"
                    
                    # 라벨에 따른 색상 변경
                    if cls_id == 2 or "Red" in label_name:
                        color = (0, 0, 255)   # 빨간색
                    elif cls_id == 1 or "Cube" in label_name:
                        color = (255, 165, 0) # 파란 육면체는 주황 박스로 표시(눈에 띄게)
                    else:
                        color = (0, 255, 0)   # 원기둥은 초록색
                        
                    cv2.rectangle(combined_view, (x1, y1), (x2, y2), color, 2)
                    info_str = f"{label_name} {conf:.2f} (R:{aspect_ratio:.1f})"
                    cv2.putText(combined_view, info_str, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 2. 비전 엣지 뷰 (우측 상단 윤곽선)
            edges = cv2.Canny(gray_clahe, 50, 150)
            edge_mini = cv2.resize(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), (240, 135))
            combined_view[20:155, 700:940] = edge_mini
            cv2.rectangle(combined_view, (700, 20), (940, 155), (0, 255, 0), 2)

            # 3. 캡처 및 DB/API 통신
            if capture_requested:
                capture_count += 1
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                file_name = f"cap_{ts}_D{detection_info}.jpg"
                path = os.path.join(SAVE_FOLDER, file_name)
                
                # 이미지 파일 저장
                cv2.imwrite(path, frame_resized)
                
                db_msg = "DB:FAIL"
                try:
                    conn = psycopg2.connect(**DB_CONFIG)
                    cur = conn.cursor()
                    cur.execute("INSERT INTO image_logs (file_path, brightness) VALUES (%s, %s)", (path, lux))
                    conn.commit()
                    cur.close(); conn.close()
                    db_msg = "DB:SAVED"
                except: pass

                api_msg = "API:WAIT"
                try:
                    # 네트워크 오류 시 화면 멈춤을 방지하기 위해 timeout=0.5 설정
                    res = requests.post(API_URL, params={"value": capture_count}, timeout=0.5)
                    api_msg = f"API:{res.status_code}"
                except:
                    api_msg = "API:TIMEOUT"

                last_log_msg = f"[CAPTURE #{capture_count}] {db_msg} | {api_msg} | {file_name}"
                log_expiry_time = time.time() + 3.0
                capture_requested = False
                
                # 캡처 시 화면 번쩍임 효과
                combined_view = cv2.bitwise_not(combined_view)

            # 4. HUD 및 상태 정보 출력
            curr_fps_time = time.time()
            fps = 1 / (curr_fps_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_fps_time
            cv2.putText(combined_view, f"FPS: {int(fps)} | LUX: {lux} | AI: {MODEL_PATH}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 저장 로그 실시간 출력 (3초간)
            if time.time() < log_expiry_time:
                cv2.putText(combined_view, last_log_msg, (20, 450), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if combined_view is not None:
            # 5. UI 버튼 렌더링 (CAPTURE 버튼 추가)
            btn_data = [(30, (0, 200, 0), "PLAY"), (180, (0, 140, 255), "PAUSE"), 
                        (330, (50, 50, 220), "EXIT"), (480, (200, 50, 200), "CAPTURE")]
            for x, color, label in btn_data:
                cv2.rectangle(combined_view, (x, 470), (x+130, 530), color, -1)
                cv2.putText(combined_view, label, (x+15, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow(window_name, combined_view)

        # 단축키 설정 (Q: 종료, C: 캡처)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('c'): capture_requested = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
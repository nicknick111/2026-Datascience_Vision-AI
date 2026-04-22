import os
# OpenCV가 내부적으로 출력하는 불필요한 경고 메시지나 로그를 숨겨서 콘솔창을 깨끗하게 유지합니다.
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

import cv2
import time
import datetime
import numpy as np
import requests
import threading
import pymysql              # 로컬 MySQL DB 접속을 위한 라이브러리
from ultralytics import YOLO # YOLOv8 AI 모델 사용을 위한 라이브러리

# =====================================================================
# [구역 1] 전역 설정 및 환경 변수
# =====================================================================
# [1순위] 원격 API 서버의 MySQL 데이터베이스 정보
REMOTE_DB_HOST = "192.168.3.21"
REMOTE_DB_USER = "root" # 실제 API 서버 DB 계정으로 변경 필요
REMOTE_DB_PASS = "1234" # 실제 API 서버 DB 비밀번호로 변경 필요
REMOTE_DB_NAME = "dronedb" # 실제 API 서버 DB 이름으로 변경 필요

# [2순위] 내 PC 로컬 MySQL 데이터베이스 정보 (백업용)
LOCAL_DB_HOST = "127.0.0.22"
LOCAL_DB_USER = "root"
LOCAL_DB_PASS = "1234"
LOCAL_DB_NAME = "dronedb"

# UI 상태 및 카메라 전역 변수
is_paused = False        # 화면 일시정지 상태
is_running = True        # 프로그램 실행 상태
use_glare_filter = True  # 빛 반사 억제(CLAHE) 필터 켜짐/꺼짐
cap = None               # 카메라 객체

# 폐루프(Closed-Loop) 재학습을 위한 미확인 이미지 저장 폴더
REVIEW_DIR = "./review_images"
os.makedirs(REVIEW_DIR, exist_ok=True)

# =====================================================================
# [구역 2] 데이터베이스 & API 통신 (Fallback 시스템)
# =====================================================================
def setup_local_database():
    """
    [목표] 내 PC의 MySQL에 접속하여 '첫 번째 검수 결과'를 저장할 테이블을 미리 만듭니다.
    """
    try:
        conn = pymysql.connect(host=LOCAL_DB_HOST, user=LOCAL_DB_USER, password=LOCAL_DB_PASS, port=3306)
        cur = conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {LOCAL_DB_NAME}")
        conn.commit()
        
        conn.select_db(LOCAL_DB_NAME)
        # 첫 번째 검수 결과 Table 생성 (존재하지 않는다면)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS first_inspection_results (
                inspection_id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME NOT NULL,
                product_name VARCHAR(100) NOT NULL,
                yolo_confidence FLOAT NOT NULL,
                anomaly_score FLOAT NOT NULL,
                status VARCHAR(50) NOT NULL
            );
        """)
        conn.commit()
        conn.close()
        print("✅ [DB] 로컬 데이터베이스(dronedb) 및 테이블 세팅 완료.")
    except Exception as e:
        print(f"❌ [DB 오류] 로컬 DB 설정 실패: {e}")

def send_inspection_result(class_name, conf, anomaly_score, status):
    """
    [핵심 통신 로직 - 수정됨] 
    1. 우선 API 서버의 원격 MySQL DB로 직접 데이터 저장을 시도합니다.
    2. 원격 DB 접속이 불가능하면, 내 PC의 로컬 MySQL(dronedb)에 직접 접속하여 데이터를 백업 저장합니다.
    (이 함수는 메인 화면이 끊기지 않도록 항상 '백그라운드 스레드'에서 몰래 실행됩니다.)
    """
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # [1단계] API 서버 원격 MySQL DB 저장 시도
    try:
        # 원격 접속이므로 타임아웃을 1초로 짧게 주어 화면 지연(렉)을 방지합니다.
        conn = pymysql.connect(
            host=REMOTE_DB_HOST, user=REMOTE_DB_USER, 
            password=REMOTE_DB_PASS, database=REMOTE_DB_NAME, 
            port=3306, connect_timeout=1
        )
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO first_inspection_results 
            (timestamp, product_name, yolo_confidence, anomaly_score, status) 
            VALUES (%s, %s, %s, %s, %s)
        """, (current_time, class_name, float(conf), float(anomaly_score), status))
        conn.commit()
        conn.close()
        print(f"🌐 [API 서버 DB 저장 성공] {class_name} -> {status}")
        return # 성공했으므로 로컬 DB 로직은 건너뜁니다.
    except Exception:
        pass # 원격 DB 접속 실패 시 당황하지 않고 조용히 아래 로컬 DB 저장 단계로 넘어갑니다.

    # [2단계] 원격 DB 실패 시 내 PC(Local) DB에 직접 저장 (Fallback 백업)
    try:
        conn = pymysql.connect(
            host=LOCAL_DB_HOST, user=LOCAL_DB_USER, 
            password=LOCAL_DB_PASS, database=LOCAL_DB_NAME, port=3306
        )
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO first_inspection_results 
            (timestamp, product_name, yolo_confidence, anomaly_score, status) 
            VALUES (%s, %s, %s, %s, %s)
        """, (current_time, class_name, float(conf), float(anomaly_score), status))
        conn.commit()
        conn.close()
        print(f"💾 [내 PC 로컬 DB 저장 성공] API 서버 연결 실패로 백업됨: {class_name} -> {status}")
    except Exception as e:
        print(f"❌ [통신 완전 실패] 원격 및 로컬 DB 모두 연결할 수 없습니다: {e}")

# =====================================================================
# [구역 3] 하이브리드 AI 모델 (YOLO + 이상 탐지)
# =====================================================================
class AnomalyDetector:
    """
    [다단계 파이프라인] YOLO가 제품을 찾으면, 이 클래스가 해당 제품의 미세 불량을 판정합니다.
    (현재는 원리 이해를 돕기 위한 가상 점수 부여기로 작동합니다.)
    """
    def inspect(self, cropped_image):
        # 0.0(완벽 정상) ~ 1.0(심각한 불량) 사이의 점수를 시뮬레이션
        score = np.random.uniform(0.0, 1.0)
        return score

# =====================================================================
# [구역 4] UI 제어 및 영상 처리 함수 (OpenCV)
# =====================================================================
def mouse_callback(event, x, y, flags, param):
    """마우스 클릭 위치(좌표)를 감지하여 하단 UI 버튼(PLAY, PAUSE, EXIT)을 작동시킵니다."""
    global is_paused, is_running
    if event == cv2.EVENT_LBUTTONDOWN:
        if 680 <= y <= 750:
            if 40 <= x <= 180:      is_paused = False  # PLAY
            elif 220 <= x <= 360:   is_paused = True   # PAUSE
            elif 400 <= x <= 540:   is_running = False # EXIT

def apply_clahe(image):
    """[최적화] 화면의 빛 반사를 억제하고 어두운 그림자를 밝혀 AI 인식률을 높이는 필터입니다."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def on_trackbar(val): pass

def set_exposure(val):
    """렌즈의 하드웨어 노출값을 물리적으로 조절합니다 (-10 ~ 0)."""
    global cap
    if cap is not None and cap.isOpened():
        cap.set(cv2.CAP_PROP_EXPOSURE, val - 10)

# =====================================================================
# [구역 5] 메인 실행 루프 (프로그램의 심장)
# =====================================================================
def main():
    global is_paused, is_running, use_glare_filter, cap
    
    # 1. 시스템 초기화 (로컬 DB 세팅 및 모델 로드)
    setup_local_database()
    
    print("⏳ AI 모델(YOLOv8n)을 불러오는 중입니다...")
    # 실제 학습된 가중치(best.pt)가 없다면 기본 모델(yolov8n.pt)로 작동합니다.
    model_path = "best.pt" if os.path.exists("best.pt") else "yolov8n.pt"
    yolo_model = YOLO(model_path)
    anomaly_detector = AnomalyDetector()

    # 2. 카메라 연결 및 설정
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ 카메라를 찾을 수 없습니다.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) # 자동 노출 끄기

    # 3. UI 윈도우 및 트랙바 생성
    window_name = 'Vision AI 하이브리드 검수 시스템'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.createTrackbar('S/W Brightness (-100~100)', window_name, 100, 200, on_trackbar)
    cv2.createTrackbar('Lens Exposure (-10~0)', window_name, 2, 10, set_exposure)
    set_exposure(2)

    # UI 도장판(Cache) 미리 만들어두기 (속도 최적화)
    base_canvas = np.zeros((798, 1152, 3), dtype=np.uint8)
    cv2.rectangle(base_canvas, (0, 648), (1152, 798), (35, 35, 35), -1)
    cv2.rectangle(base_canvas, (40, 680), (180, 750), (0, 200, 0), -1)      # PLAY
    cv2.putText(base_canvas, "PLAY", (75, 720), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.rectangle(base_canvas, (220, 680), (360, 750), (0, 140, 255), -1)   # PAUSE
    cv2.putText(base_canvas, "PAUSE", (245, 720), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.rectangle(base_canvas, (400, 680), (540, 750), (50, 50, 200), -1)   # EXIT
    cv2.putText(base_canvas, "EXIT", (440, 720), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    prev_time = 0
    last_send_time = 0

    print("✅ 모든 준비 완료. 영상 검수를 시작합니다!")

    # ====== 실시간 루프 시작 ======
    while is_running:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        if not is_paused:
            ret, frame = cap.read()
            if not ret: break

            # 캔버스 크기에 맞게 1차 리사이즈
            frame_resized = cv2.resize(frame, (1152, 648))

            # [영상 보정] 트랙바에 따른 밝기 조절
            actual_brightness = cv2.getTrackbarPos('S/W Brightness (-100~100)', window_name) - 100 
            if actual_brightness > 0:
                bright_arr = np.full(frame_resized.shape, actual_brightness, dtype=np.uint8)
                filtered_frame = cv2.add(frame_resized, bright_arr)
            elif actual_brightness < 0:
                bright_arr = np.full(frame_resized.shape, abs(actual_brightness), dtype=np.uint8)
                filtered_frame = cv2.subtract(frame_resized, bright_arr)
            else:
                filtered_frame = frame_resized.copy()

            # [영상 보정] 단축키 'f' 필터 적용
            if use_glare_filter:
                filtered_frame = apply_clahe(filtered_frame)
                
            # [1차 Vision AI 검사] 빛 반사가 제거된 깨끗한 이미지로 YOLO 객체 인식 수행!
            results = yolo_model(filtered_frame, conf=0.5, verbose=False)

            # [최적화] API/DB 전송은 3초에 한 번만 이루어지도록 쿨다운 설정
            curr_time = time.time()
            can_send_db = (curr_time - last_send_time >= 3)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = yolo_model.names[class_id]

                    # 예외 방지: 박스가 화면 밖으로 나가지 않게 고정
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(1152, x2), min(648, y2)

                    # [다단계 파이프라인] YOLO가 찾은 영역만 잘라내어 이상 탐지기(Anomaly)에 전달
                    cropped_product = filtered_frame[y1:y2, x1:x2]
                    
                    if cropped_product.size > 0:
                        anomaly_score = anomaly_detector.inspect(cropped_product)

                        # [품질 관리 판정 기준]
                        if anomaly_score <= 0.3:
                            status = "PASS"
                            color = (0, 255, 0) # 초록색
                        elif anomaly_score >= 0.7:
                            status = "FAIL"
                            color = (0, 0, 255) # 빨간색
                        else:
                            status = "REVIEW_NEEDED"
                            color = (0, 255, 255) # 노란색
                            # [폐루프] 재학습을 위해 애매한 이미지는 폴더에 자동 저장
                            cv2.imwrite(f"{REVIEW_DIR}/review_{int(time.time()*1000)}.jpg", cropped_product)

                        # 화면에 결과 그리기 (박스와 텍스트)
                        cv2.rectangle(filtered_frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{class_name} | A-Score: {anomaly_score:.2f} | {status}"
                        cv2.putText(filtered_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # [DB/API 전송] 3초 쿨다운이 찼다면, 멈춤 없이 백그라운드 스레드로 전송
                        if can_send_db:
                            threading.Thread(
                                target=send_inspection_result, 
                                args=(class_name, conf, anomaly_score, status), 
                                daemon=True
                            ).start()
                            last_send_time = curr_time

            # 화면 구성을 위해 combined_view 업데이트
            combined_view = filtered_frame

            # FPS 계산
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time

        # -----------------------------------------------------
        # 화면 최종 병합 및 출력
        # -----------------------------------------------------
        if 'combined_view' in locals() and combined_view is not None:
            canvas = base_canvas.copy() # 변하지 않는 UI 도장판 복사
            canvas[0:648, 0:1152] = combined_view # 위에 카메라 화면 덮어쓰기

            # 실시간 변동 텍스트 그리기
            try: display_exposure = cv2.getTrackbarPos('Lens Exposure (-10~0)', window_name) - 10
            except: display_exposure = -8
            
            cv2.putText(canvas, f"LIVE | FPS: {int(fps)}", (780, 685), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(canvas, f"Brightness: {actual_brightness}", (780, 715), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
            cv2.putText(canvas, f"Lens Exposure: {display_exposure}", (780, 745), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 255), 2)
            cv2.putText(canvas, f"Filter(f): {'ON' if use_glare_filter else 'OFF'}", (780, 775), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255) if use_glare_filter else (150, 150, 150), 2)

            if is_paused:
                cv2.putText(canvas, "--- PAUSED ---", (400, 324), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

            cv2.imshow(window_name, canvas)

        # 키보드 입력 감지 (종료 및 필터 토글)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            use_glare_filter = not use_glare_filter

    # 프로그램 종료 시 자원 안전 해제
    print("[SYSTEM] 시스템을 안전하게 종료합니다.")
    if cap is not None: cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
# =====================================================================
#  Video_Record.py (AI Vision 데이터셋 녹화기 - 최종 실무형)
#  [업데이트] Intrinsic 보정 추가, 파이프라인 UI 표시, 녹화 타이머 적용
# =====================================================================

import os
import cv2
import time
import threading
import numpy as np
from queue import Queue

# OpenCV 경고 메시지 숨김
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

# =====================================================================
# [환경 설정 변수]
# =====================================================================
DISPLAY_W, DISPLAY_H = 640, 480 
CANVAS_W, CANVAS_H = 1000, 580   
is_running = True                
is_recording = False             
rec_start_time = 0               # 녹화 시작 시간을 기록할 변수

record_queue = Queue(maxsize=120) 
save_dir = "./datasets"          

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# =====================================================================
# [Step 1] Intrinsic 캘리브레이션 파라미터 (로이체 RPC-20F 실제 측정값)
# =====================================================================
MTX = np.array([
    [922.75191915,   0.0,         349.17192687], 
    [  0.0,         925.73078326, 283.64098051], 
    [  0.0,           0.0,          1.0       ]
], dtype=np.float64)

DIST = np.array([
    [5.16538290e-01, -5.82044664e+00, 2.25062615e-02, 3.80271478e-03, 2.57957417e+01]
], dtype=np.float64)

# =====================================================================
# [Step 2] AI 전처리 파이프라인 (Intrinsic + Filters)
# =====================================================================
def advanced_preprocessing(frame):
    # 1. Intrinsic Undistortion (렌즈 기하학적 왜곡 펴기)
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(MTX, DIST, (w,h), 0, (w,h))
    undistorted = cv2.undistort(frame, MTX, DIST, None, newcameramtx)

    # 2. Guided Filter (빛 반사 및 잔노이즈 부드럽게 뭉개기)
    guided = cv2.ximgproc.guidedFilter(guide=undistorted, src=undistorted, radius=5, eps=50)

    # 3. CLAHE (어두운 디테일 끌어올리기 - ClipLimit 1.5로 안정화)
    lab = cv2.cvtColor(guided, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)) 
    cl = clahe.apply(l)
    processed_lab = cv2.merge((cl, a, b))
    processed_bgr = cv2.cvtColor(processed_lab, cv2.COLOR_LAB2BGR)
    
    return processed_bgr

# =====================================================================
# [Step 3] 백그라운드 비디오 저장 일꾼 (하드디스크 쓰기 전담)
# =====================================================================
def record_worker():
    video_writer = None
    current_path = ""

    while is_running or not record_queue.empty():
        if not record_queue.empty():
            path, frame = record_queue.get() 
            
            if video_writer is None or current_path != path:
                if video_writer is not None:
                    video_writer.release()
                current_path = path
                video_writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'avc1'), 20.0, (DISPLAY_W, DISPLAY_H))
            
            video_writer.write(frame)
        else:
            time.sleep(0.01) 

    if video_writer is not None:
        video_writer.release()

threading.Thread(target=record_worker, daemon=True).start()

# =====================================================================
# [Step 4] 화면 우측 상태창(UI) 그리기 함수
# =====================================================================
def draw_ui(canvas, fps, hl_ratio):
    
    x_offset = DISPLAY_W + 20
    color_alert = (0, 0, 255) if hl_ratio > 10 else (0, 255, 0)
    
    # 1. 시스템 상태
    cv2.putText(canvas, "[ Vision AI Recorder ]", (x_offset, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(canvas, f"FPS       : {fps:.1f}", (x_offset, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(canvas, f"Highlight : {hl_ratio:.2f}%", (x_offset, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_alert, 2)
    
    cv2.line(canvas, (x_offset, 140), (CANVAS_W - 20, 140), (100, 100, 100), 1)
    
    # 2. 적용 중인 전처리 파이프라인 표시
    cv2.putText(canvas, "[ Applied Pipeline ]", (x_offset, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
    cv2.putText(canvas, "1. Intrinsic Undistort", (x_offset, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(canvas, "2. Guided Filter (eps:50)", (x_offset, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(canvas, "3. CLAHE (clip:1.5)", (x_offset, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(canvas, "4. ROI Highlight Check", (x_offset, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.line(canvas, (x_offset, 330), (CANVAS_W - 20, 330), (100, 100, 100), 1)

    # 3. 조작법 안내
    cv2.putText(canvas, "[ Controls ]", (x_offset, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.putText(canvas, "R: Record Start/Stop", (x_offset, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(canvas, "Q: Quit Program", (x_offset, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# =====================================================================
# [Step 5] 메인 루프 (카메라 구동 및 UI 처리)
# =====================================================================
def main():
    global is_running, is_recording, rec_start_time

    cap = None
    for i in range(2):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) # 윈도우 설정창 띄우기 위해 CAP_DSHOW 사용
        if cap.isOpened():
            print(f"✅ 카메라 연결 성공 (인덱스: {i})")
            cap.set(cv2.CAP_PROP_SETTINGS, 1)    # 윈도우 카메라 밝기 제어창 호출
            break

    if cap is None or not cap.isOpened():
        print("❌ Error: 카메라를 열 수 없습니다.")
        return

    current_rec_path = ""
    prev_time = time.time()

    while is_running:
        ret, frame = cap.read()
        if not ret: break

        # 1. AI 학습용 순수 이미지 전처리
        processed = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
        processed = advanced_preprocessing(processed)

        # 2. 상태 검사 (ROI 빛 반사 체크)
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[150:350, 100:540] # 주황색 부품 영역만 검사
        hl_ratio = float(np.mean(roi_gray >= 245)) * 100.0           
        
        # 3. FPS 계산
        fps = 1 / (time.time() - prev_time) if (time.time() - prev_time) > 0 else 0
        prev_time = time.time()

        # 4. 관리자용 껍데기(UI) 씌우기
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[0:DISPLAY_H, 0:DISPLAY_W] = processed
        
        
        
        # 우측 패널 그리기
        draw_ui(canvas, fps, hl_ratio)

        # ------------------------------------------------------------------
        # 5. 비디오 녹화 & 타이머 로직
        # ------------------------------------------------------------------
        if is_recording:
            if rec_start_time == 0: 
                rec_start_time = time.time()
                
            if current_rec_path == "":
                current_rec_path = os.path.join(save_dir, f"Dataset_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
            
            # 녹화 큐에 순수 이미지 던지기
            if not record_queue.full():
                record_queue.put((current_rec_path, processed.copy()))
            
            # 타이머 계산 (MM:SS)
            elapsed_sec = int(time.time() - rec_start_time)
            mins, secs = divmod(elapsed_sec, 60)
            timer_str = f"{mins:02d}:{secs:02d}"
            
            # UI에 깜빡이는 빨간 점과 타이머 표시
            if int(time.time() * 2) % 2 == 0:
                cv2.circle(canvas, (700, 500), 10, (0, 0, 255), -1)
            cv2.putText(canvas, f"REC {timer_str}", (730, 510), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
        else:
            rec_start_time = 0
            current_rec_path = "" 
            cv2.putText(canvas, "STANDBY", (720, 510), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # 6. 화면 출력
        cv2.imshow("Vision AI Monitor", canvas)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):     
            is_running = False
        elif key == ord('r'):   
            is_recording = not is_recording
            if is_recording: print("🟢 녹화를 시작합니다...")
            else: print("🔴 녹화를 종료합니다.")

    if cap: cap.release()
    cv2.destroyAllWindows()
    print("시스템이 안전하게 종료되었습니다.")

if __name__ == "__main__":
    main()
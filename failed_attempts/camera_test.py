import os
import cv2
import time
import numpy as np

# 로그 정리
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

is_paused = False
is_running = True

def mouse_callback(event, x, y, flags, param):
    """하단 UI 마우스 클릭 제어"""
    global is_paused, is_running
    if event == cv2.EVENT_LBUTTONDOWN:
        if 500 <= y <= 560:
            if 20 <= x <= 120:      is_paused = False
            elif 140 <= x <= 240:   is_paused = True
            elif 260 <= x <= 360:   is_running = False

def analyze_frame_metrics(frame_bgr):
    """
    핵심 품질 지표(명암비, 선명도)만 백그라운드에서 빠르게 계산합니다.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # ① 명암비 차이 (Otsu 이진화 기반 자동 배경/객체 분리)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bg_mean = cv2.mean(gray, mask=cv2.bitwise_not(mask))[0]
    obj_mean = cv2.mean(gray, mask=mask)[0]
    contrast_gap = abs(obj_mean - bg_mean)
    
    # ② 엣지 선명도 (Laplacian 분산 - 초점 및 모션블러 확인)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return contrast_gap, sharpness

def main():
    global is_paused, is_running
    os.system('cls' if os.name == 'nt' else 'clear')
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Camera not found.")
        return

    # 카메라 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = 'Quality Inspection Monitor - Final Clean View'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    # UI 캔버스 (가로 1000px: 메인 화면 640px + 정보 패널 360px)
    base_canvas = np.zeros((580, 1000, 3), dtype=np.uint8)
    
    cv2.rectangle(base_canvas, (640, 0), (1000, 480), (30, 30, 30), -1) # 우측 패널
    cv2.rectangle(base_canvas, (0, 480), (1000, 580), (40, 40, 40), -1) # 하단 패널
    
    # 하단 컨트롤 버튼
    cv2.rectangle(base_canvas, (20, 500), (120, 560), (0, 180, 0), -1)
    cv2.putText(base_canvas, "PLAY", (45, 538), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(base_canvas, (140, 500), (240, 560), (0, 120, 255), -1)
    cv2.putText(base_canvas, "PAUSE", (160, 538), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(base_canvas, (260, 500), (360, 560), (50, 50, 180), -1)
    cv2.putText(base_canvas, "EXIT", (290, 538), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    prev_time = 0

    while is_running:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        if not is_paused:
            ret, frame = cap.read()
            if not ret: break

            frame_resized = cv2.resize(frame, (640, 480))
            
            # 백그라운드에서 지표만 빠르게 계산
            gap, sharpness = analyze_frame_metrics(frame_resized)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time

            canvas = base_canvas.copy()
            
            # 1. 메인 화면에 컬러 원본 전체 출력 (하얀 박스 및 텍스트 오버레이 제거됨)
            canvas[0:480, 0:640] = frame_resized  

            # 2. 우측 대시보드 텍스트 출력
            cv2.putText(canvas, "[1] Contrast Gap", (660, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(canvas, f"Gap Value : {gap:.1f}", (660, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(canvas, "Target    : > 30~50", (660, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            cv2.putText(canvas, "Meaning   : YOLO Accuracy", (660, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

            cv2.putText(canvas, "[2] Edge Sharpness", (660, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(canvas, f"Laplacian : {sharpness:.1f}", (660, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(canvas, "Target    : Monitor for Blur", (660, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            cv2.putText(canvas, "Meaning   : Focus Status", (660, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

            cv2.putText(canvas, "[3] Real-time FPS", (660, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.putText(canvas, f"Current   : {fps:.1f} FPS", (660, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
            cv2.putText(canvas, "Capacity  : > 10 FPS needed", (660, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            if is_paused:
                cv2.putText(canvas, "--- PAUSED ---", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            cv2.imshow(window_name, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
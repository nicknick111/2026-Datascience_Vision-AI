import os
import cv2
import time
import numpy as np

# 로그 정리 (불필요한 콘솔 출력 방지)
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

is_paused = False
is_running = True
use_filter = True # 품질 개선 필터 기본 활성화

def mouse_callback(event, x, y, flags, param):
    """하단 UI 마우스 클릭 제어"""
    global is_paused, is_running
    if event == cv2.EVENT_LBUTTONDOWN:
        if 500 <= y <= 560:
            if 20 <= x <= 120:      is_paused = False
            elif 140 <= x <= 240:   is_paused = True
            elif 260 <= x <= 360:   is_running = False

def on_exposure_trackbar(val):
    """카메라 하드웨어 렌즈 노출값 직접 제어"""
    global cap
    if cap is not None and cap.isOpened():
        exposure_val = val - 10 
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure_val)

def apply_image_enhancement(frame):
    """
    [최종 보정 팁 반영] RPC-20F 어두운 환경 극복을 위한 최적화 전처리
    """
    # 1. 미세 노이즈 억제 (커널 크기 상향: 3x3 -> 5x5)
    # 어두운 영역의 자글거리는 센서 노이즈를 더 부드럽게 뭉개줍니다.
    img = cv2.GaussianBlur(frame, (5, 5), 0)

    # 2. 컬러 밸런스 유지 (HSV 채도 약한 부스팅: +15)
    # 색수차(보라색/파란색 띠)를 방지하기 위해 채도를 과하지 않게 살짝만 올립니다.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, 15) 
    hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 3. 대비 강도 미세 조절 (CLAHE clipLimit: 4.0 -> 3.5)
    # 내부 부품의 그림자는 밝히되, 너무 인위적인 느낌이 나지 않도록 조절합니다.
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    img = cv2.merge((cl, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    # 4. 감마 보정을 통한 중간톤(갈색/회색) 밝기 향상
    gamma = 1.5
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img, table)

    # 5. 윤곽선 선명도 복원 (Unsharp Masking)
    # 블러로 인해 살짝 뭉개진 테두리를 다시 날카롭게 깎아냅니다.
    blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
    img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

    return img

def main():
    global is_paused, is_running, use_filter, cap
    os.system('cls' if os.name == 'nt' else 'clear')
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Camera not found.")
        return

    # 카메라 하드웨어 캡처 해상도 (성능과 시야각 밸런스)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 하드웨어 수동 노출 설정 (자동 노출 기능 끄기)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25) 

    window_name = 'RPC-20F Quality Enhancer v8.1'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    # 렌즈 노출 조절 트랙바 추가
    cv2.createTrackbar('Lens Exposure (-10~0)', window_name, 6, 10, on_exposure_trackbar)
    on_exposure_trackbar(6) # 기본값 -4로 세팅

    # UI 구성을 위한 캔버스 캐싱
    base_canvas = np.zeros((580, 640, 3), dtype=np.uint8)
    cv2.rectangle(base_canvas, (0, 480), (640, 580), (40, 40, 40), -1)
    
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

            # 연산 부하를 최소화하기 위해 640x480으로 리사이즈 후 처리
            frame_resized = cv2.resize(frame, (640, 480))

            if use_filter:
                display_frame = apply_image_enhancement(frame_resized)
            else:
                display_frame = frame_resized.copy()

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time

            # 최종 화면 구성
            canvas = base_canvas.copy()
            canvas[0:480, 0:640] = display_frame
            
            # 우측 하단 상태 정보 출력
            f_status = "ON" if use_filter else "OFF"
            try:
                display_exposure = cv2.getTrackbarPos('Lens Exposure (-10~0)', window_name) - 10
            except:
                display_exposure = "Auto"

            cv2.putText(canvas, f"FPS: {int(fps)}", (400, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(canvas, f"Lens Exp: {display_exposure}", (400, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
            cv2.putText(canvas, f"Quality Filter(f): {f_status}", (400, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            if is_paused:
                cv2.putText(canvas, "--- PAUSED ---", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            cv2.imshow(window_name, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('f'): use_filter = not use_filter

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
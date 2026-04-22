import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

import cv2
import time
import datetime
import numpy as np

is_paused = False
is_running = True

def mouse_callback(event, x, y, flags, param):
    global is_paused, is_running
    if event == cv2.EVENT_LBUTTONDOWN:
        if 470 <= y <= 530:
            if 30 <= x <= 180:      
                is_paused = False
            elif 210 <= x <= 360:   
                is_paused = True
            elif 390 <= x <= 540:   
                is_running = False

def main():
    global is_paused, is_running
    os.system('cls' if os.name == 'nt' else 'clear')
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Camera not found.")
        return

    # 카메라 원본 화질(해상도) 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=" * 60)
    print("[SYSTEM] RPC-20F Vision Studio v2.3 is running.")
    print("[INFO] Basic Camera Mode (Auto-lighting Disabled)")
    print("=" * 60)

    window_name = 'RPC-20F Vision Studio v2.3'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    prev_time = 0
    combined_view = None

    while is_running:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        if not is_paused:
            ret, frame = cap.read()
            if not ret: break

            # UI와 마우스 클릭 위치가 틀어지지 않도록 화면 표시 크기는 960x540으로 고정 유지합니다.
            frame_resized = cv2.resize(frame, (960, 540))

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time

            # [핵심 변경 사항] 자동 밝기 조절 없이, 카메라 원본 이미지를 그대로 사용합니다.
            combined_view = frame_resized.copy()

            # HUD 정보 표시 (FPS만 표시하고, 조도 관련 텍스트는 제거했습니다)
            cv2.putText(combined_view, f"LIVE | FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if combined_view is not None:
            # UI 버튼 그리기
            cv2.rectangle(combined_view, (30, 470), (180, 530), (0, 200, 0), -1)
            cv2.putText(combined_view, "PLAY", (70, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.rectangle(combined_view, (210, 470), (360, 530), (0, 140, 255), -1)
            cv2.putText(combined_view, "PAUSE", (245, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.rectangle(combined_view, (390, 470), (540, 530), (50, 50, 200), -1)
            cv2.putText(combined_view, "EXIT", (435, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if is_paused:
                cv2.putText(combined_view, "--- PAUSED ---", (650, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)

            cv2.imshow(window_name, combined_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("[SYSTEM] App closed safely.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
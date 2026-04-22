import os
import cv2
import time
import numpy as np
import threading
from flask import Flask, Response, jsonify
from flask_cors import CORS  # 추가
from ultralytics import YOLO


# =====================================================================
# [1] 시스템 초기화 및 Flask 설정
# =====================================================================
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
app = Flask(__name__)
CORS(app) # 추가: 웹 프론트엔드에서의 외부 접근 허용

# AI 모델 가중치 경로 (부트캠프 PC 환경에 맞춰 설정)
BEST_PT_PATH = r"E:\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\best.pt"

# 화면 크기 설정 (640x480은 YOLOv8n 권장 입력 해상도와 일치함)
DISPLAY_W, DISPLAY_H = 640, 480
CANVAS_W, CANVAS_H = 1000, 580

# 전처리 필터 초기 상태
USE_AUTO_EXPOSURE = True  # 자동 노출 보정
USE_CLAHE = True          # 대비 제한 히스토그램 평활화
USE_GAUSSIAN = True       # 노이즈 제거
USE_EDGE_VIEW = False     # 윤곽선 보기 모드

is_paused = False
is_running = True

# =====================================================================
# [2] 멀티스레드 동기화 설정 (Flask 웹서버와 UI 공유용)
# =====================================================================
global_frame = None  # 모든 스레드가 공유하는 최신 화면
global_result = {"status": "WAITING", "confidence": 0.0, "class": "None"}
lock = threading.Lock() # 동시 접근 시 충돌 방지용 잠금장치

# =====================================================================
# [3] 영상 전처리 엔진 (품질 개선용)
# =====================================================================
def adjust_gamma(image, gamma=1.0):
    """감마값이 1.0보다 크면 화면이 어두워지며 대비가 강화됨"""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def auto_exposure_control(frame_bgr, target_mean=115, high_sat_ratio_th=0.08):
    """하이라이트 10% 유지를 위한 정밀 밝기 제어"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_val, highlight_ratio = float(gray.mean()), float(np.mean(gray >= 245))
    out = frame_bgr.copy()
    
    if highlight_ratio > high_sat_ratio_th:
        out = adjust_gamma(out, gamma=1.45) 
        
    gray2 = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    current_mean = float(gray2.mean())
    
    if current_mean > target_mean + 5:
        out = cv2.convertScaleAbs(out, alpha=0.92, beta=-20) 
        
    return out, mean_val, highlight_ratio

def apply_clahe_l_channel(frame_bgr, clip=2.0, tile=(8, 8)):
    """부품의 디테일과 질감을 강조하는 필터"""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile).apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

def analyze_frame_metrics(frame_bgr):
    """Contrast와 Sharpness 수치를 계산하여 품질 점수 산출"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bg_mean = cv2.mean(gray, mask=cv2.bitwise_not(mask))[0]
    obj_mean = cv2.mean(gray, mask=mask)[0]
    return abs(obj_mean - bg_mean), cv2.Laplacian(gray, cv2.CV_64F).var()

def get_quality_status(contrast_gap, sharpness):
    """품질 점수에 따른 상태 메시지 결정"""
    if contrast_gap >= 130.0 and sharpness >= 700.0: return "GOOD", (0, 220, 0)
    elif contrast_gap >= 110 and sharpness >= 500: return "NORMAL", (0, 200, 255)
    else: return "CHECK", (0, 0, 255)

# =====================================================================
# [4] UI 가시성 보정 및 제어 함수
# =====================================================================
def draw_static_ui():
    base_canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    cv2.rectangle(base_canvas, (640, 0), (1000, 480), (30, 30, 30), -1)
    cv2.rectangle(base_canvas, (0, 480), (1000, 580), (40, 40, 40), -1)
    
    cv2.rectangle(base_canvas, (40, 500), (240, 560), (0, 180, 0), -1)
    cv2.putText(base_canvas, "PLAY", (110, 538), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.rectangle(base_canvas, (260, 500), (460, 560), (0, 120, 255), -1)
    cv2.putText(base_canvas, "PAUSE", (320, 538), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.rectangle(base_canvas, (480, 500), (680, 560), (50, 50, 180), -1)
    cv2.putText(base_canvas, "EXIT", (550, 538), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return base_canvas

def draw_filter_states(canvas):
    """우측 하단 필터 작동 정보 표시"""
    font, scale, th, color = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1, (200, 200, 200)
    cv2.putText(canvas, f"AUTO(A): {'ON' if USE_AUTO_EXPOSURE else 'OFF'}", (720, 520), font, scale, color, th)
    cv2.putText(canvas, f"CLAHE(C): {'ON' if USE_CLAHE else 'OFF'}", (860, 520), font, scale, color, th)
    cv2.putText(canvas, f"GAUSS(G): {'ON' if USE_GAUSSIAN else 'OFF'}", (720, 550), font, scale, color, th)
    cv2.putText(canvas, f"EDGE(E): {'ON' if USE_EDGE_VIEW else 'OFF'}", (860, 550), font, scale, color, th)

def draw_dynamic_text(canvas, contrast, sharpness, fps, status, color, mean, highlight, ai_class, ai_conf, ai_status):
    """매 프레임 갱신되는 센서 및 판정 데이터 출력"""
    cv2.putText(canvas, "[Camera Metrics]", (660, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(canvas, f"Contrast : {contrast:.1f}", (660, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(canvas, f"Sharpness: {sharpness:.1f}", (660, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(canvas, f"Mean Lux : {mean:.1f}", (660, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, f"HighLight: {highlight * 100:.1f}%", (660, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(canvas, f"Cam Status: {status}", (660, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.line(canvas, (650, 240), (990, 240), (100, 100, 100), 1)
    cv2.putText(canvas, "[AI Inspection Result]", (660, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    s_color = (0, 255, 255) if ai_status == "INSPECTING..." else (0, 255, 0) if ai_status == "DETECTED" else (100, 100, 255)
    cv2.putText(canvas, f"Target: {ai_class}", (660, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)
    cv2.putText(canvas, f"Conf  : {ai_conf:.1f}%", (660, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)
    cv2.putText(canvas, f"STATUS: {ai_status}", (660, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.8, s_color, 2)
    cv2.putText(canvas, f"FPS: {fps:.1f}", (660, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)

# =====================================================================
# [5] 카메라 & AI 처리 메인 루프 (실시간 판정 적용)
# =====================================================================
def camera_yolo_loop():
    global global_frame, global_result, is_running, is_paused
    global USE_AUTO_EXPOSURE, USE_CLAHE, USE_GAUSSIAN, USE_EDGE_VIEW
    
    try:
        model = YOLO(BEST_PT_PATH)
    except Exception as e: print(f"AI Loading Fail: {e}"); os._exit(1)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): print("Camera Open Fail"); os._exit(1)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # 수동 노출 모드 고정
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)     # 밝기 확보
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "Unified AI Vision Monitor"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    base_canvas = draw_static_ui()
    prev_time = 0
    display_class, display_conf, display_status = "None", 0.0, "WAITING"

    while is_running:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break
        if not is_paused:
            ret, frame = cap.read()
            if not ret: break

            # [전처리 파이프라인]
            frame_resized = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            if USE_AUTO_EXPOSURE:
                proc_frame, mean_val, high_ratio = auto_exposure_control(frame_resized)
            else:
                gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                mean_val, high_ratio, proc_frame = float(gray.mean()), float(np.mean(gray >= 245)), frame_resized.copy()
            
            if USE_CLAHE: proc_frame = apply_clahe_l_channel(proc_frame)
            if USE_GAUSSIAN: proc_frame = cv2.GaussianBlur(proc_frame, (3,3), 0)

            if USE_EDGE_VIEW:
                edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY), (3,3), 0), 70, 150)
                display_frame = proc_frame.copy()
                display_frame[edges > 0] = (0, 255, 255) # 노란색 윤곽선
            else: display_frame = proc_frame.copy()

            contrast, sharpness = analyze_frame_metrics(display_frame)
            status, color = get_quality_status(contrast, sharpness)

            # [AI 실시간 판정]
            results = model.predict(source=display_frame, imgsz=640, conf=0.6, iou=0.45, device='0', verbose=False)
            annotated_frame = results[0].plot()

            if len(results[0].boxes) > 0:
                best = results[0].boxes[0]
                display_class = model.names[int(best.cls[0])]
                display_conf = float(best.conf[0]) * 100
                display_status = "DETECTED"
            else:
                display_class = "None"
                display_conf = 0.0
                display_status = "WAITING"
                
            with lock: 
                global_result = {"status": display_status, "confidence": round(display_conf, 2), "class": display_class}

           # UI 렌더링 및 FPS 계산
            fps = 1 / (time.time() - prev_time) if prev_time > 0 else 0; prev_time = time.time()
            canvas = base_canvas.copy()
            canvas[0:DISPLAY_H, 0:DISPLAY_W] = annotated_frame
            draw_dynamic_text(canvas, contrast, sharpness, fps, status, color, mean_val, high_ratio, display_class, display_conf, display_status)
            draw_filter_states(canvas)

            with lock: global_frame = canvas.copy()
            cv2.imshow(window_name, canvas)

        # [키보드 제어 생략 - 기존과 동일]
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): is_running = False
        elif key == ord('p'): is_paused = not is_paused
        elif key == ord('a'): USE_AUTO_EXPOSURE = not USE_AUTO_EXPOSURE
        elif key == ord('c'): USE_CLAHE = not USE_CLAHE
        elif key == ord('g'): USE_GAUSSIAN = not USE_GAUSSIAN
        elif key == ord('e'): USE_EDGE_VIEW = not USE_EDGE_VIEW

    cap.release(); cv2.destroyAllWindows()

def mouse_callback(event, x, y, flags, param):
    global is_paused, is_running
    if event == cv2.EVENT_LBUTTONDOWN and 500 <= y <= 560:
        if 40 <= x <= 240: is_paused = False
        elif 260 <= x <= 460: is_paused = True
        elif 480 <= x <= 680: is_running = False

# =====================================================================
# [6] Flask 웹 서버 인터페이스
# =====================================================================
@app.route('/video_feed')
def video_feed():
    def generate():
        while is_running:
            with lock:
                if global_frame is None: continue
                ret, jpeg = cv2.imencode('.jpg', global_frame)
            if ret: yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/result')
def get_result():
    with lock: return jsonify(global_result)

@app.route('/')
def index(): return '<h1>Unified AI Vision System</h1><img src="/video_feed" width="1000" height="580">'

if __name__ == '__main__':
    threading.Thread(target=camera_yolo_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
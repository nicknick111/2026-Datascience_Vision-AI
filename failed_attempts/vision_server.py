import os
import cv2
import time
import numpy as np
import threading
from flask import Flask, Response, jsonify
from ultralytics import YOLO

# =====================================================================
# [1] 기본 설정 및 Flask 초기화
# =====================================================================
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
app = Flask(__name__)

# AI 모델 가중치 파일 경로 (사용자 환경에 맞게 수정)
BEST_PT_PATH = r"best.pt" 

DISPLAY_W, DISPLAY_H = 640, 480
CANVAS_W, CANVAS_H = 1000, 580

USE_AUTO_EXPOSURE = True
USE_CLAHE = True
USE_GAUSSIAN = True
USE_EDGE_VIEW = False

# =====================================================================
# [2] 스레드 동기화 및 공유 변수
# =====================================================================
global_frame = None
global_result = {"status": "WAIT", "confidence": 0.0, "class": "None"}
lock = threading.Lock()

# =====================================================================
# [3] 영상 전처리 및 UI 그리기 함수 
# =====================================================================
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def auto_exposure_control(frame_bgr, target_mean=118, high_sat_ratio_th=0.08):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_val, highlight_ratio = float(gray.mean()), float(np.mean(gray >= 245))
    out = frame_bgr.copy()
    if highlight_ratio > high_sat_ratio_th:
        out = adjust_gamma(out, gamma=1.35)
    gray2 = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    if float(gray2.mean()) > target_mean + 10:
        out = cv2.convertScaleAbs(out, alpha=0.92, beta=-18)
    return out, mean_val, highlight_ratio

def apply_clahe_l_channel(frame_bgr, clip=2.0, tile=(8, 8)):
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile).apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

def analyze_frame_metrics(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bg_mean = cv2.mean(gray, mask=cv2.bitwise_not(mask))[0]
    obj_mean = cv2.mean(gray, mask=mask)[0]
    return abs(obj_mean - bg_mean), cv2.Laplacian(gray, cv2.CV_64F).var()

def get_quality_status(contrast_gap, sharpness):
    if contrast_gap >= 130.0 and sharpness >= 700.0: return "GOOD", (0, 220, 0)
    elif contrast_gap >= 110 and sharpness >= 500: return "NORMAL", (0, 200, 255)
    else: return "CHECK", (0, 0, 255)

def draw_static_ui():
    base_canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    cv2.rectangle(base_canvas, (640, 0), (1000, 480), (30, 30, 30), -1)
    cv2.rectangle(base_canvas, (0, 480), (1000, 580), (40, 40, 40), -1)
    cv2.putText(base_canvas, "WEB STREAMING ACTIVE | TRIGGER MODE", (40, 538), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return base_canvas

def draw_dynamic_text(canvas, contrast, sharpness, fps, status, color, mean, highlight, ai_class, ai_conf, ai_status):
    cv2.putText(canvas, "[Camera Metrics]", (660, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(canvas, f"Contrast : {contrast:.1f}", (660, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(canvas, f"Sharpness: {sharpness:.1f}", (660, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(canvas, f"Mean Lux : {mean:.1f}", (660, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, f"HighLight: {highlight * 100:.1f}%", (660, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(canvas, f"Cam Status: {status}", (660, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.putText(canvas, "[AI Inspection Result]", (660, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # 상태별 색상 분기
    if ai_status == "INSPECTING...":
        status_color = (0, 255, 255) # 노란색
    elif ai_status == "DETECTED":
        status_color = (0, 255, 0)   # 녹색
    else:
        status_color = (100, 100, 255) # 빨간색

    cv2.putText(canvas, f"Target: {ai_class}", (660, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)
    cv2.putText(canvas, f"Conf  : {ai_conf:.1f}%", (660, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 100), 2)
    cv2.putText(canvas, f"STATUS: {ai_status}", (660, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(canvas, f"FPS: {fps:.1f}", (660, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)

# =====================================================================
# [4] 백그라운드 AI 처리 스레드 (가상 트리거 10초 주기 + 10회 검사 적용)
# =====================================================================
def camera_yolo_loop():
    global global_frame, global_result
    
    try:
        model = YOLO(BEST_PT_PATH)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] AI 모델(best.pt)을 불러올 수 없습니다: {e}")
        os._exit(1)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    # [기능 추가 1] 카메라를 찾지 못하면 즉시 오류 메시지 출력 후 서버 전체 종료
    if not cap.isOpened():
        print("\n" + "="*60)
        print("[CRITICAL ERROR] 카메라를 찾을 수 없거나 연결되지 않았습니다.")
        print(" -> 웹캠이 PC에 정상적으로 연결되어 있는지, 다른 프로그램이 사용 중인지 확인하세요.")
        print(" -> 서버를 강제 종료합니다.")
        print("="*60 + "\n")
        os._exit(1) # 메인 스레드인 Flask까지 모두 강제 종료

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    base_canvas = draw_static_ui()
    prev_time = 0

    # [기능 추가 2] 가상 트리거용 제어 변수
    last_trigger_time = time.time()
    inspection_attempts_left = 0
    temp_results = [] # 10번 시도 동안의 임시 결과 저장소
    
    # UI에 고정적으로 띄울 이전 검사 결과 상태값
    display_class = "None"
    display_conf = 0.0
    display_status = "WAITING TRIGGER"

    print("[INFO] 10초 주기 가상 트리거 기반 모니터링이 시작되었습니다.")

    while True:
        ret, frame = cap.read()
        if not ret: continue

        frame_resized = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
        
        if USE_AUTO_EXPOSURE:
            frame_resized, mean_val, highlight_ratio = auto_exposure_control(frame_resized)
        else:
            gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            mean_val, highlight_ratio = float(gray.mean()), float(np.mean(gray >= 245))
            
        if USE_CLAHE: frame_resized = apply_clahe_l_channel(frame_resized)
        if USE_GAUSSIAN: frame_resized = cv2.GaussianBlur(frame_resized, (3,3), 0)

        contrast_gap, sharpness = analyze_frame_metrics(frame_resized)
        cam_status, cam_color = get_quality_status(contrast_gap, sharpness)

        # -----------------------------------------------------------------
        # 트리거 시간 체크 (10초마다 1회 신호 발생)
        # -----------------------------------------------------------------
        current_time = time.time()
        if current_time - last_trigger_time >= 10.0:
            print(f"\n[TRIGGER] 10초 경과! AI 식별 10회 시도를 시작합니다... ({time.strftime('%H:%M:%S')})")
            last_trigger_time = current_time
            inspection_attempts_left = 10
            temp_results = []
            display_status = "INSPECTING..." # 검사 중임을 알리는 상태

        # -----------------------------------------------------------------
        # AI 식별 로직 (트리거 신호를 받고 횟수가 남아있을 때만 작동)
        # -----------------------------------------------------------------
        if inspection_attempts_left > 0:
            results = model.predict(source=frame_resized, conf=0.6, device='cpu', verbose=False)
            annotated_frame = results[0].plot()

            if len(results[0].boxes) > 0:
                best_box = results[0].boxes[0]
                cls_name = model.names[int(best_box.cls[0])]
                conf_val = float(best_box.conf[0]) * 100
                temp_results.append((cls_name, conf_val))
            else:
                temp_results.append(("None", 0.0))

            inspection_attempts_left -= 1

            # 10회 시도가 방금 완전히 끝난 순간 -> 결과 종합 (다수결 또는 최고 신뢰도 기준)
            if inspection_attempts_left == 0:
                valid_detections = [res for res in temp_results if res[0] != "None"]

                if valid_detections:
                    # 10회 중 한 번이라도 인식했다면, 가장 신뢰도가 높았던 값을 최종 확정
                    best_result = max(valid_detections, key=lambda x: x[1])
                    display_class = best_result[0]
                    display_conf = best_result[1]
                    display_status = "DETECTED"
                    print(f" -> [완료] 식별 성공: {display_class} ({display_conf:.1f}%)")
                else:
                    # 10회 내내 인식 실패
                    display_class = "None"
                    display_conf = 0.0
                    display_status = "FAIL (NOT FOUND)"
                    print(" -> [완료] 객체를 식별하지 못했습니다.")

                # 외부 관리자 페이지(/result)에 넘길 글로벌 결과 업데이트
                with lock:
                    global_result = {
                        "status": display_status,
                        "confidence": round(display_conf, 2),
                        "class": display_class
                    }
        else:
            # 검사 횟수가 끝났거나 대기 중일 때는 AI 연산을 건너뛰고 화면만 송출 (자원 최적화)
            annotated_frame = frame_resized.copy()

        # UI 캔버스 구성
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time

        canvas = base_canvas.copy()
        canvas[0:DISPLAY_H, 0:DISPLAY_W] = annotated_frame
        draw_dynamic_text(canvas, contrast_gap, sharpness, fps, cam_status, cam_color, 
                          mean_val, highlight_ratio, display_class, display_conf, display_status)

        # 락 안에서 안전하게 전역 프레임 교체
        with lock:
            global_frame = canvas.copy()

# =====================================================================
# [5] Flask 웹 스트리밍 라우팅
# =====================================================================
def generate_mjpeg():
    while True:
        with lock:
            if global_frame is None:
                continue
            ret, jpeg = cv2.imencode('.jpg', global_frame)
            
        if not ret: continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.03)

@app.route('/')
def index():
    return '<h1>AI Vision Streaming</h1><img src="/video_feed" width="1000" height="580">'

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/result', methods=['GET'])
def get_result():
    with lock:
        return jsonify(global_result)

if __name__ == '__main__':
    print("[INFO] AI 비전 처리 스레드 시작...")
    t = threading.Thread(target=camera_yolo_loop, daemon=True)
    t.start()

    print("[INFO] Flask 웹 서버를 0.0.0.0:8080 에서 시작합니다.")
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
import os
import cv2
import time
import numpy as np

# 로그 정리 (OpenCV 경고 메시지 억제)
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

# =====================================================================
# [1] 시스템 전역 상태 및 경로 설정
# =====================================================================
is_paused = False
is_running = True
is_recording = False
video_writer = None
save_dir = "recorded_videos"

record_start_time = 0
current_save_path = ""

# 화면 / UI 크기 설정 (AI 모델 입력 규격인 640x480 유지)
DISPLAY_W, DISPLAY_H = 640, 480
CANVAS_W, CANVAS_H = 1000, 580

# ---------------------------------------------------------
# [NEW] Intrinsic 캘리브레이션 데이터 로드
# ---------------------------------------------------------
try:
    # intrinsic_calibration.py에서 생성된 npy 파일을 불러옵니다.
    calib_data = np.load('camera_calib_data.npy', allow_pickle=True).item()
    K = calib_data['mtx']
    D = calib_data['dist']
    print("[System] ✅ Camera Calibration Data Loaded.")
except Exception as e:
    print(f"[Warning] ⚠️ Calibration Data not found: {e}")
    K, D = None, None

# =====================================================================
# [2] 전처리 필터 파라미터 (unified_vision_system.py와 완벽 동기화)
# =====================================================================
USE_AUTO_EXPOSURE = True  # 자동 노출 보정 활성화
USE_CLAHE = True          # 대비 제한 히스토그램 평활화 (질감 강조)
USE_GAUSSIAN = True       # 노이즈 제거
USE_EDGE_VIEW = False     # 윤곽선 보기 모드

CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
GAUSSIAN_KERNEL = (3, 3)

# [수정] 과노출 판단 기준: 검은 여백 제거 후 더욱 정밀한 제어를 위해 0.04(4%)로 하향
TARGET_MEAN_BRIGHTNESS = 115
HIGHLIGHT_RATIO_THRESHOLD = 0.04  # 하이라이트가 4%만 넘어도 밝기 억제

# 품질 기준 (UI 표시용)
GOOD_CONTRAST_MIN = 130.0
GOOD_SHARPNESS_MIN = 700.0


# =====================================================================
# [3] 영상 전처리 엔진 함수
# =====================================================================
def adjust_gamma(image, gamma=1.0):
    """감마 보정: 1.0보다 크면 어두운 영역의 대비가 강화됨"""
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def auto_exposure_control(frame_bgr, target_mean=115, high_sat_ratio_th=0.04):
    """하이라이트 억제를 포함한 지능형 밝기 제어"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_val = float(gray.mean())
    highlight_ratio = float(np.mean(gray >= 245)) # 245 이상의 밝은 픽셀 비율

    out = frame_bgr.copy()

    # 1. 하이라이트(빛 번짐)가 기준치 초과 시 감마 보정으로 강제 억제
    if highlight_ratio > high_sat_ratio_th:
        out = adjust_gamma(out, gamma=1.45)

    # 2. 평균 밝기가 목표치보다 높을 경우 선형적으로 어둡게 조정
    gray2 = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    current_mean = float(gray2.mean())

    if current_mean > target_mean + 5:
        # 밝기를 8% 줄이고(alpha), 전체적으로 20만큼 어둡게(beta) 만듭니다.
        out = cv2.convertScaleAbs(out, alpha=0.92, beta=-20)

    return out, mean_val, highlight_ratio

def apply_clahe_l_channel(frame_bgr, clip=2.0, tile=(8, 8)):
    """LAB 색공간의 L(밝기) 채널에만 CLAHE를 적용하여 색 왜곡 없이 질감만 강조"""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile).apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

# =====================================================================
# [4] 핵심 전처리 파이프라인 (왜곡 보정 + ROI 크롭 포함)
# =====================================================================
def preprocess_frame(frame_bgr):
    global USE_AUTO_EXPOSURE, USE_CLAHE, USE_GAUSSIAN, USE_EDGE_VIEW
    global K, D

    # 1. 기본 리사이즈 (640x480)
    frame_resized = cv2.resize(frame_bgr, (DISPLAY_W, DISPLAY_H))

    # 2. [NEW] Intrinsic 왜곡 보정 및 검은 여백(Black Border) 제거
    if K is not None and D is not None:
        h, w = frame_resized.shape[:2]
        # alpha=0: 왜곡을 펴면서 생기는 검은 여백을 모두 잘라낸 '유효 영역(ROI)'을 계산합니다.
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0, (w, h))
        
        # 렌즈 왜곡 물리적 보정 실행
        undistorted = cv2.undistort(frame_resized, K, D, None, newcameramtx)
        
        # 계산된 ROI 영역으로 화면을 잘라내어(Crop) 검은색 테두리를 완전히 제거합니다.
        x, y, w_roi, h_roi = roi
        if w_roi > 0 and h_roi > 0:
            undistorted = undistorted[y:y+h_roi, x:x+w_roi]
            # 잘라낸 후 작아진 화면을 다시 640x480으로 복원하여 AI 입력 규격을 유지합니다.
            proc_frame = cv2.resize(undistorted, (DISPLAY_W, DISPLAY_H))
        else:
            proc_frame = undistorted
    else:
        proc_frame = frame_resized.copy()

    # 3. 자동 노출 보정 (보정된 프레임 기준)
    if USE_AUTO_EXPOSURE:
        proc_frame, mean_val, highlight_ratio = auto_exposure_control(
            proc_frame, target_mean=TARGET_MEAN_BRIGHTNESS, high_sat_ratio_th=HIGHLIGHT_RATIO_THRESHOLD)
    else:
        gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
        mean_val, highlight_ratio = float(gray.mean()), float(np.mean(gray >= 245))

    # 4. 질감 강조(CLAHE) 및 노이즈 제거(Gaussian)
    if USE_CLAHE: proc_frame = apply_clahe_l_channel(proc_frame, clip=CLAHE_CLIP, tile=CLAHE_TILE)
    if USE_GAUSSIAN: proc_frame = cv2.GaussianBlur(proc_frame, GAUSSIAN_KERNEL, 0)

    # 5. 윤곽선 모드 (엣지 추출)
    if USE_EDGE_VIEW:
        gray_blur = cv2.GaussianBlur(cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY), (3,3), 0)
        edges = cv2.Canny(gray_blur, 70, 150)
        display_frame = proc_frame.copy()
        display_frame[edges > 0] = (0, 255, 255) # 노란색 윤곽선 표시
    else: 
        display_frame = proc_frame.copy()

    return display_frame, mean_val, highlight_ratio

# =====================================================================
# [5] 품질 지표 및 UI 관련 함수 (분석용)
# =====================================================================
def analyze_frame_metrics(frame_bgr):
    """Contrast와 Sharpness 수치를 계산하여 이미지 품질 점수 산출"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bg_mean = cv2.mean(gray, mask=cv2.bitwise_not(mask))[0]
    obj_mean = cv2.mean(gray, mask=mask)[0]
    contrast_gap = abs(obj_mean - bg_mean)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return contrast_gap, sharpness

def get_quality_status(contrast_gap, sharpness):
    if contrast_gap >= GOOD_CONTRAST_MIN and sharpness >= GOOD_SHARPNESS_MIN: return "GOOD", (0, 220, 0)
    elif contrast_gap >= 110 and sharpness >= 500: return "NORMAL", (0, 200, 255)
    else: return "CHECK", (0, 0, 255)

def draw_static_ui():
    """버튼 등 고정된 UI 캔버스 생성"""
    base_canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    cv2.rectangle(base_canvas, (640, 0), (1000, 480), (30, 30, 30), -1)
    cv2.rectangle(base_canvas, (0, 480), (1000, 580), (40, 40, 40), -1)

    cv2.rectangle(base_canvas, (20, 500), (120, 560), (0, 180, 0), -1)
    cv2.putText(base_canvas, "PLAY", (45, 538), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(base_canvas, (140, 500), (240, 560), (0, 120, 255), -1)
    cv2.putText(base_canvas, "PAUSE", (160, 538), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(base_canvas, (260, 500), (360, 560), (100, 100, 100), -1)
    cv2.putText(base_canvas, "RECORD", (275, 538), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.rectangle(base_canvas, (380, 500), (480, 560), (50, 50, 180), -1)
    cv2.putText(base_canvas, "EXIT", (410, 538), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return base_canvas

def draw_dynamic_text(canvas, contrast_gap, sharpness, fps, status_text, status_color, mean_val, highlight_ratio):
    cv2.putText(canvas, f"Contrast: {contrast_gap:.1f}", (660, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(canvas, f"Sharpness: {sharpness:.1f}", (660, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(canvas, f"FPS: {fps:.1f}", (660, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)
    cv2.putText(canvas, f"Mean Lux: {mean_val:.1f}", (660, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, f"HighLight: {highlight_ratio * 100:.1f}%", (660, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(canvas, f"Cam Status: {status_text}", (660, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

def draw_filter_states(canvas):
    global USE_AUTO_EXPOSURE, USE_CLAHE, USE_GAUSSIAN, USE_EDGE_VIEW
    y0, font, scale, th = 455, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
    cv2.putText(canvas, f"AUTO(A): {'ON' if USE_AUTO_EXPOSURE else 'OFF'}", (650, y0), font, scale, (200, 200, 200), th)
    cv2.putText(canvas, f"CLAHE(C): {'ON' if USE_CLAHE else 'OFF'}", (790, y0), font, scale, (200, 200, 200), th)
    cv2.putText(canvas, f"GAUSS(G): {'ON' if USE_GAUSSIAN else 'OFF'}", (650, y0 + 18), font, scale, (200, 200, 200), th)
    cv2.putText(canvas, f"EDGE(E): {'ON' if USE_EDGE_VIEW else 'OFF'}", (790, y0 + 18), font, scale, (200, 200, 200), th)

def mouse_callback(event, x, y, flags, param):
    global is_paused, is_running, is_recording, video_writer
    global record_start_time, current_save_path

    if event == cv2.EVENT_LBUTTONDOWN:
        if 500 <= y <= 560:
            if 20 <= x <= 120: is_paused = False
            elif 140 <= x <= 240: is_paused = True
            elif 260 <= x <= 360:
                is_recording = not is_recording
                if is_recording:
                    os.makedirs(save_dir, exist_ok=True)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    relative_path = os.path.join(save_dir, f"inspection_{timestamp}.mp4")
                    current_save_path = os.path.abspath(relative_path)
                    record_start_time = time.time()
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(current_save_path, fourcc, 10.0, (CANVAS_W, CANVAS_H))
                    print(f"\n[▶] RECORDING STARTED -> {current_save_path}\n")
                else:
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    record_start_time = 0
                    print(f"\n[■] RECORDING STOPPED & SAVED -> {current_save_path}\n")
            elif 380 <= x <= 480: is_running = False

# =====================================================================
# [6] 메인 루프 (실행부)
# =====================================================================
def main():
    global is_paused, is_running, is_recording, video_writer
    global record_start_time
    global USE_AUTO_EXPOSURE, USE_CLAHE, USE_GAUSSIAN, USE_EDGE_VIEW

    os.system('cls' if os.name == 'nt' else 'clear')

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Camera not found.")
        return

    # [하드웨어 제어] AI 비전 시스템과 동일한 카메라 노출 설정 고정
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # 수동 노출 모드 고정
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)     # 물리적 노출값 강제 저하 (-5)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "Quality Inspection Data Collector"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    base_canvas = draw_static_ui()
    prev_time = 0
    frozen_canvas = base_canvas.copy()

    while is_running:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break

        if not is_paused:
            ret, frame = cap.read()
            if not ret: break

            # [핵심] 통합 전처리 파이프라인 적용 (왜곡 보정 및 필터링)
            processed_frame, mean_val, highlight_ratio = preprocess_frame(frame)

            # 영상 품질 지표 계산
            contrast_gap, sharpness = analyze_frame_metrics(processed_frame)
            status_text, status_color = get_quality_status(contrast_gap, sharpness)

            fps = 1 / (time.time() - prev_time) if prev_time > 0 else 0; prev_time = time.time()

            # UI 합성
            canvas = base_canvas.copy()
            canvas[0:DISPLAY_H, 0:DISPLAY_W] = processed_frame
            draw_dynamic_text(canvas, contrast_gap, sharpness, fps, status_text, status_color, mean_val, highlight_ratio)
            draw_filter_states(canvas)

            # 녹화 중일 경우 프레임 저장
            if is_recording:
                elapsed = time.time() - record_start_time
                time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                cv2.rectangle(canvas, (260, 500), (360, 560), (0, 0, 220), -1)
                cv2.putText(canvas, f"REC {time_str}", (270, 538), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                if video_writer is not None: video_writer.write(canvas)
            
            frozen_canvas = canvas.copy()
        else:
            # 일시 정지 화면 표시
            canvas = frozen_canvas.copy()
            cv2.putText(canvas, "--- PAUSED ---", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow(window_name, canvas)

        # 키보드 단축키 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('p'): is_paused = not is_paused
        elif key == ord('a'): USE_AUTO_EXPOSURE = not USE_AUTO_EXPOSURE
        elif key == ord('c'): USE_CLAHE = not USE_CLAHE
        elif key == ord('g'): USE_GAUSSIAN = not USE_GAUSSIAN
        elif key == ord('e'): USE_EDGE_VIEW = not USE_EDGE_VIEW

    cap.release()
    if video_writer is not None: video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
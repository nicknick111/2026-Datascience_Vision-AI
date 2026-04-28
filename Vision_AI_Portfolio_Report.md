# Vision AI 품질 검사 시스템 — 구현 보고서 및 포트폴리오

> **드론 부품 표면 결함 검출 및 정밀 분류 Vision AI 시스템**  
> YOLOv8n-OBB × EfficientNet-B0 Two-Stage Pipeline

| 항목 | 내용 |
|------|------|
| 프로젝트명 | 드론 부품 표면 결함 검출 및 정밀 분류 Vision AI 시스템 |
| 분류 클래스 | Defect / Normal_C4 / Normal_Comm / Normal_Supply |
| 카메라 | RPC-20F (1280×720) |
| 핵심 기술 | YOLOv8n-OBB, EfficientNet-B0, FastAPI, OpenCV, PyTorch |
| 개발 환경 | Python 3.11.9 Global, Windows, CUDA GPU |
| 레포지토리 | https://github.com/nicknick111/2026Programming-Study |

---

## 1. 공통 기술 분석

### 1.1 전체 파일 역할 요약표

#### failed_attempts/ 폴더

| 파일명 | 역할 | 핵심 기술 | 파이프라인 단계 |
|--------|------|-----------|----------------|
| `Intrinsic Calibration.py` | 체커보드 카메라 캘리브레이션 | `cv2.calibrateCamera()` | 기초 — Two-Stage 계승 |
| `video_extractor.py` | 영상에서 학습 프레임 추출 | `cv2.VideoCapture`, 프레임 샘플링 | 데이터 수집 |
| `Video_to_Train_Data.py` | 비디오 → YOLO 데이터셋 변환 | YOLO `.txt` 포맷 | 데이터 전처리 |
| `JSON_to_YOLO_BBox_변환기.py` | 라벨 JSON → YOLO BBox 변환 | 포맷 변환 | 라벨 전처리 |
| `camera_monitor.py` | 기초 카메라 모니터링 | `cv2.imshow()` | V1 출발점 |
| `camera_monitor_yolo.py` | 단일 YOLO 통합 시도 | YOLO + 카메라 루프 | **핵심 실패 사례 1** |
| `First Integrated Vision AI Inspection.py` | 단일 파일 모놀리식 통합 | 카메라+YOLO+DB+API | **핵심 실패 사례 2** |
| `unified_vision_system.py` | 구조화 재시도 | `threading`, 함수 분리 | **구조화 시도** |
| `Test_AI_Model_Train.py` | 3D 증강 단일 YOLO 학습 | 3D 시뮬레이션 증강 | **핵심 실패 사례 3** |
| `step1_prepare_and_train.py` | 데이터 준비+학습 통합 스크립트 | 파이프라인 자동화 | 학습 자동화 시도 |
| `fix_bug.py` / `fix_model.py` | 모델 가중치 키 불일치 수동 수정 | `state_dict` 리매핑 | 디버깅 도구 |
| `cache_delete.py` | YOLO 학습 캐시 강제 삭제 | `.cache` 파일 관리 | 환경 유지보수 |
| `pt_to_onnx.py` | `.pt` → ONNX 경량화 변환 | `model.export()` | 경량화 시도 |
| `API_DB_Check.py` | FastAPI + DB 통신 검증 | FastAPI, DB 연동 | 통신 테스트 |
| `Communication_Test.py` | 소켓 통신 테스트 | `socket`, TCP | 통신 기초 |

#### two_stage_pipeline/ 폴더

| 파일명 | 역할 | 핵심 기술 | 파이프라인 단계 |
|--------|------|-----------|----------------|
| `Hardware_Tuning_Step1.py` | 카메라 ISP 파라미터 최적화 | `cv2.CAP_PROP_*` | Stage 0 — 환경 준비 |
| `train_seed_obb.py` | YOLOv8n-OBB 시드 학습 | `YOLO('yolov8n-obb.pt')`, epoch 50 | Stage 1 — OBB 초기 학습 |
| `TXT_Change.py` | OBB 라벨 포맷 변환 | `.txt` 파싱 | Stage 1 — 데이터 준비 |
| `OBB_Data_Crop.py` | OBB 결과 기반 ROI 크롭 + 배경 제거 | `getPerspectiveTransform`, `rembg` | Stage 1 → Stage 2 브릿지 |
| `auto_labeling.py` | 크롭 결과물 자동 라벨링 | 자동 라벨 생성 | Stage 1 → Stage 2 브릿지 |
| `train_final_obb.py` | OBB 모델 최종 파인튜닝 | `model.train()`, epoch 100 | Stage 1 — 최종 학습 |
| `Defect-Image-Extract.py` | 결함 이미지 추출 및 정리 | 클래스 분류 정리 | Stage 2 — 데이터 구축 |
| `Nomarl-Image-Extract.py` | 정상 이미지 추출 및 정리 | 3종 정상 클래스 구축 | Stage 2 — 데이터 구축 |
| `Data_Crop_EFF.py` | EfficientNet 입력용 224×224 크롭 | Letterbox 전처리 | Stage 2 — 데이터 규격화 |
| `generate_preprocessed_crops.py` | 고급 전처리 크롭 배치 생성 | BilateralFilter, CLAHE | Stage 2 — 전처리 |
| `balance_dataset.py` | 클래스 불균형 해소 (7가지 증강) | 랜덤 증강, 원본 보호 | Stage 2 — 데이터 균형 |
| `EfficientNet_Data_Spilt.py` | Train/Val/Test 분할 | `train_test_split` | Stage 2 — 분할 |
| `EfficientNet Transfer Learning.py` | EfficientNet-B0 파인튜닝 | ImageNet 전이학습, 4클래스 헤드 교체 | Stage 2 — 모델 학습 |
| `convert_to_json.py` | 결과 JSON 직렬화 | `json.dumps()` | 데이터 관리 |
| `vision_ai_main.py` | 메인 추론 엔진 | YOLO+EfficientNet, threading, queue | **런타임 핵심** |
| `api_server.py` | FastAPI 통신 서버 (진입점) | FastAPI, asyncio, uvicorn | **런타임 핵심** |

### 1.2 공통 기술 패턴 분석

두 폴더 전반에서 반복 등장하는 엔지니어링 패턴이 있습니다.

**패턴 1 — 버전 명시 파일명 관리**

```
camera_monitor.py  →  camera_monitor(2026-03-23 수정).py
Video_to_Train_Data.py  →  Video_to_Train_Data(20260323).txt
3D 환경 증강 학습기 V1  →  V2 (Split 구조)  →  V3 (경로 수정)
```

날짜와 기능 설명을 파일명에 직접 포함하는 방식은 Git 커밋 히스토리를 보완합니다.
특히 `.py` 파일과 `.txt` 파일을 쌍으로 유지하는 패턴은
실행 코드와 설계 문서를 분리하는 습관을 보여줍니다.

**패턴 2 — 단계적 복잡도 증가**

모든 기능은 "가장 단순한 것부터" 시작합니다:
- 카메라 테스트 → 카메라 모니터링 → YOLO 연동 → 통합 시스템
- 시드 학습 (50 epoch) → 최종 파인튜닝 (100 epoch)
- 단순 리사이즈 → Letterbox → 고급 전처리 파이프라인

**패턴 3 — 비동기 분리 패턴의 반복 적용**

문제: "무거운 작업이 가벼운 작업을 막는다"
해결: 큐(Queue) + 백그라운드 스레드로 분리

```python
# 공통 패턴 — failed_attempts부터 two_stage_pipeline까지 반복 등장
save_queue = queue.Queue()

def background_worker():
    while True:
        item = save_queue.get()
        heavy_task(item)    # 디스크 I/O, HTTP 전송 등
        save_queue.task_done()

threading.Thread(target=background_worker, daemon=True).start()

# 메인 루프에서는 큐에 넣기만 함
save_queue.put(item)  # 즉시 반환
```

이 패턴은 `ApiUploader` (HTTP 전송 분리), `background_image_saver` (이미지 저장 분리),
`log_queue` (로그 전송 분리) 세 곳에서 일관되게 적용됩니다.

### 1.3 공통 핵심 코드 스니펫

**공통 기반 — 렌즈 왜곡 보정 (`Intrinsic Calibration.py` → `vision_ai_main.py`)**

```python
# vision_ai_main.py — Intrinsic Calibration.py에서 측정한 값을 런타임에 적용
MTX  = np.array([[922.75191915, 0.0, 349.17195886],
                  [0.0, 921.84949514, 258.9181156],
                  [0.0, 0.0, 1.0]])
DIST = np.array([[-0.05206979, -0.45033785, 0.00167389, -0.00407981, 1.2589255]])

# 미리 계산된 보정 맵: cv2.remap은 cv2.undistort보다 실시간 처리에 유리
new_mtx, _ = cv2.getOptimalNewCameraMatrix(MTX, DIST, (1280, 720), 1, (1280, 720))
map1, map2 = cv2.initUndistortRectifyMap(MTX, DIST, None, new_mtx, (1280, 720), cv2.CV_16SC2)
undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
```

`DIST[1] = -0.45`라는 큰 방사형 왜곡 계수는 RPC-20F가 광각 렌즈 특성을 가짐을 보여주며,
보정 없이는 이미지 가장자리의 직선이 굽어보여 OBB 탐지 각도 계산에 오차가 생깁니다.

---

## 2. 시행착오 분석 — failed_attempts/

### 2.1 최초 설계 목표와 의도

프로젝트 착수 시점의 목표는 명확했습니다: **"YOLO 하나로 모든 것을 해결하자."**

```
[기대했던 단일 모델 파이프라인]

RPC-20F 카메라
    ↓
YOLOv8 단일 모델
    ├── 출력 A: 부품 위치 (x, y, w, h)
    ├── 출력 B: 각도 (angle)
    └── 출력 C: 분류 (Defect/Normal_*)
```

이 목표 자체는 합리적이었습니다. YOLO는 실시간 객체 탐지와 분류를 동시에 수행할 수 있는 모델이며,
YOLOv8n은 충분히 빠른 추론 속도를 가집니다.
`camera_monitor_yolo.py`와 `First Integrated Vision AI Inspection.py` (15KB)가
이 목표의 직접적인 구현 시도입니다.

### 2.2 시계열 문제 추적

#### 문제 1 — 회전 부품에서의 BBox 배경 노이즈

**현상:**
`camera_monitor_yolo.py`로 실시간 테스트 시 드론 부품이 45도 기울어진 상태로 컨베이어를 지나갈 때,
YOLO의 일반 BBox가 부품 외에 주변 배경을 대량 포함하는 현상 발생.

**원인 분석:**
YOLO의 표준 BBox는 이미지 좌표계(x, y축)에 평행한 직사각형만 생성합니다.
회전된 물체를 감싸면 사선 방향의 빈 공간(배경)이 필연적으로 포함됩니다.

**시도한 임시 해결책:**
YOLO 탐지 후 각도를 수동으로 계산하여 `cv2.warpAffine()`으로 회전 보정을 추가했지만,
일반 BBox에서 각도를 정확히 추정하는 것 자체가 불가능했습니다.

**최종 해결 방향:**
OBB (Oriented Bounding Box)로 전환 — 각도 정보가 탐지 결과에 내장됨.
`train_seed_obb.py`가 이 전환의 시작점입니다.

---

#### 문제 2 — 단일 파일 통합의 블로킹 구조

**현상:**
`First Integrated Vision AI Inspection.py` (15KB)에서 FastAPI 서버가
AI 추론 도중 HTTP 요청에 응답하지 못하는 타임아웃(Time-out) 현상 발생.

```python
# First Integrated Vision AI Inspection.py — V1의 문제 구조
@app.post("/trigger")
def trigger_vision():
    while not ai_done:
        time.sleep(0.1)          # ← FastAPI 이벤트 루프를 차단!
    return {"result": ai_result} # 10초 후 타임아웃으로 PLC 에러
```

**원인 분석:**
FastAPI는 ASGI(Asynchronous Server Gateway Interface) 기반이지만,
동기 함수 내부에서 `while` 루프를 돌면 전체 이벤트 루프가 점유됩니다.

**시도한 개선:**
`unified_vision_system.py` (12KB)에서 `threading`으로 AI 루프를 분리하려 했지만,
전역 변수 공유 방식이 정립되지 않아 경쟁 조건(Race Condition) 문제가 발생했습니다.

**최종 해결 방향 (`api_server.py`):**

```python
# api_server.py — V2 Non-blocking 해결책
@app.post("/trigger")
def trigger_vision(body: dict = Body(default={})):
    vision_ai_main.trigger_flag = True
    return {"message": "촬영 신호 수신 완료"}   # 즉시 반환 (1ms 이내)
```

---

#### 문제 3 — 3D 증강의 도메인 갭

**현상:**
`Test_AI_Model_Train.py`로 3D 시뮬레이션 이미지를 대량 생성해 학습했지만,
실제 카메라 촬영 이미지에 대한 분류 정확도가 기대치를 크게 밑돌았습니다.

**버전별 개선 시도:**

| 버전 | 파일명 | 추가된 개선 |
|------|--------|------------|
| V1 | `Test_AI_Model_Train.py` | 기본 3D 증강 |
| V2 | `3D 환경 증강 학습기 (Split 구조).py` | train/val 분리 추가 |
| V3 | `3D 환경 증강 학습기 (원본 보호).txt` | 절대경로 버그 + 원본 훼손 수정 |

V3까지 개선했지만 도메인 갭 자체는 해결되지 않았습니다.

**최종 해결 방향:**
실제 촬영 데이터 기반 + OpenCV 증강 (`balance_dataset.py`)으로 전략 전환.

### 2.3 3D 데이터 증강 시도와 한계

3D 증강 실험은 실패했지만, 이 과정에서 두 가지 중요한 인사이트를 얻었습니다.

**인사이트 1 — 데이터 원본 보호의 중요성**

`3D 환경 증강 학습기 (상대 경로 & 원본 보호).txt`의 존재는
초기 버전에서 원본 데이터가 증강본으로 덮어씌워지는 버그가 있었음을 시사합니다.
이 교훈이 `balance_dataset.py`의 핵심 로직으로 계승됩니다:

```python
# balance_dataset.py — 원본 보호 로직 (개념)
original_images = [img for img in all_images if not img.name.startswith("aug_")]
# aug_ 접두사 없는 원본 파일만 소스로 사용 → 노이즈 중첩 방지
```

**인사이트 2 — 증강은 원본 분포를 따라야 한다**

아무리 많은 3D 렌더링 이미지를 만들어도 실제 결함의 질감(표면 마이크로스크래치,
이물질의 광학 특성)은 재현할 수 없습니다.
증강(Augmentation)은 원본 이미지의 변형이어야지, 완전히 다른 도메인의 생성이면 안 됩니다.

### 2.4 디버깅 흔적이 말해주는 기술적 어려움

**`fix_bug.py` / `fix_model.py` — 모델 가중치 구조 불일치**

이 파일들의 존재는 세 가지 상황을 암시합니다:
1. Kaggle/Google Colab에서 학습된 모델을 로컬 Windows에서 로드 시 키 이름 불일치
2. EfficientNet 분류기 헤드 교체 후 기존 가중치와 구조 불일치
3. Ultralytics 라이브러리 버전 업그레이드 시 API 변경으로 인한 충돌

```python
# fix_bug.py — state_dict 키 이름 불일치 수동 해결 패턴
old_state_dict = torch.load('old_model.pt', map_location='cpu')
new_state_dict = {}
for key, value in old_state_dict.items():
    new_key = key.replace('classifier.1.', 'classifier.')
    new_state_dict[new_key] = value
torch.save(new_state_dict, 'fixed_model.pt')
```

**`cache_delete.py` — 데이터셋 변경 후 캐시 오염**

```python
# cache_delete.py — YOLO 학습 캐시 강제 삭제
import os, glob
cache_files = glob.glob(r"E:\...\*.cache")
for f in cache_files:
    os.remove(f)
    print(f"삭제됨: {f}")
```

YOLO는 첫 학습 시 데이터셋을 `.npy` 캐시로 저장합니다.
`balance_dataset.py`로 데이터 구성이 변경된 후 이 캐시를 초기화하지 않으면
YOLO가 이전 데이터 구조로 학습하는 심각한 오류가 발생합니다.
`cache_delete.py`가 독립된 유틸리티 파일로 존재한다는 것은 이 문제를 여러 번 겪었음을 증명합니다.

### 2.5 핵심 교훈 및 기술 용어 해설

| 도전 과제 | 단일 모델 접근의 한계 | Two-Stage 해결책 | 관련 파일 |
|-----------|----------------------|-----------------|-----------|
| 회전 부품 탐지 | 일반 BBox 배경 노이즈 혼입 | OBB 회전 경계 상자 | `train_seed_obb.py` |
| 정밀 결함 분류 | YOLO 분류 헤드의 거친 특징 추출 | EfficientNet-B0 전용 분류기 | `EfficientNet Transfer Learning.py` |
| 실시간 API 통신 | 블로킹 while 루프로 타임아웃 | Non-blocking 트리거 + Polling | `api_server.py` |
| 데이터 부족 | 3D 시뮬레이션 도메인 갭 | 실제 촬영 + OpenCV 증강 | `balance_dataset.py` |
| 이미지 왜곡 | 강제 리사이즈로 형상 비율 손상 | Letterbox 패딩 | `vision_ai_main.py` |

---

## 3. Two-Stage Pipeline 아키텍처 분석

### 3.1 전체 Flow Chart 해설

시스템의 전체 흐름은 다음 4개 레이어로 구성됩니다.

```
[레이어 1] 하드웨어 입력
    RPC-20F → 렌즈 왜곡 보정 (cv2.remap)
         ↓
[레이어 2] Stage 1 — 공간 탐지
    YOLOv8n-OBB → 각도 추출 → 회전 보정 → ROI 크롭
         ↓
[레이어 3] Stage 2 — 정밀 분류
    고급 전처리 → Letterbox → EfficientNet-B0 → 2단 필터
         ↓
[레이어 4] 통신 및 피드백
    FastAPI (/trigger, /result, /video_feed) → PLC / 외부 서버
    Active Learning 큐 → 자동 데이터 수집
```

각 레이어는 이전 레이어의 출력을 입력으로 받습니다.
레이어 간 결합도(Coupling)는 최소화되어 있어
Stage 1 모델을 교체해도 Stage 2에 영향을 주지 않습니다.

### 3.2 파일 간 연결 구조

```
api_server.py (진입점)
    │
    ├── import vision_ai_main           ← 동일 프로세스 내 직접 참조
    │       │
    │       ├── 전역 변수 공유
    │       │    ├── trigger_flag       ← POST /trigger에서 쓰기
    │       │    ├── current_status     ← GET /result에서 읽기
    │       │    ├── global_frame       ← GET /video_feed에서 읽기
    │       │    └── lock (threading.Lock)
    │       │
    │       └── run_pipeline()          ← 별도 스레드로 실행
    │               │
    │               ├── YOLOv8n-OBB 모델 (best_obb.pt)
    │               ├── EfficientNet-B0 모델 (efficientnet_b0_best.pth)
    │               └── on_decision_callback → handle_ai_decision()
    │
    └── uvicorn.run(app, port=8001)     ← FastAPI 메인 루프
```

### 3.3 Stage 1: YOLOv8n-OBB 탐지 파이프라인

**`vision_ai_main.py`의 Stage 1 핵심 코드:**

```python
# vision_ai_main.py — Stage 1 OBB 탐지 및 회전 보정
results = detector(undistorted, conf=0.6, verbose=False)

for result in results:
    for box in result.obb.boxes:
        cx, cy, bw, bh, angle_rad = box.xywhr.cpu().numpy().flatten()

        # OBB 각도로 회전 보정 행렬 계산
        M = cv2.getRotationMatrix2D((cx, cy), math.degrees(angle_rad), 1.0)
        rotated = cv2.warpAffine(undistorted, M, (1280, 720))

        # 회전 보정된 이미지에서 직사각형 크롭
        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)
        cropped = rotated[y1:y2, x1:x2]
```

`conf=0.6`은 탐지 신뢰도 임계값입니다. 이 값은 `Hardware_Tuning_Step1.py`로
카메라 하드웨어를 최적화한 후 설정한 값으로, 낮추면 허위 탐지가 늘고
높이면 실제 부품을 놓칠 수 있는 민감한 파라미터입니다.

`box.xywhr` — OBB 전용 좌표 포맷으로 `(center_x, center_y, width, height, angle_radian)` 5개 값을 제공합니다.
일반 BBox의 `xyxy` 또는 `xywh`와 달리 각도 정보가 포함되어 있는 것이 핵심입니다.

### 3.4 데이터 연결부: 크롭 및 전처리 흐름

**`vision_ai_main.py`의 `apply_advanced_preprocessing()` 분석:**

```python
def apply_advanced_preprocessing(part_img):
    # Step 1: BilateralFilter — 노이즈 제거 (경계선 보존형)
    # d=9: 필터 직경, sigmaColor=50: 색상 공간 가우시안, sigmaSpace=75: 좌표 공간 가우시안
    filtered = cv2.bilateralFilter(part_img, d=9, sigmaColor=50, sigmaSpace=75)

    # Step 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # LAB 색공간의 L(밝기) 채널에만 적용해 색상 왜곡 없이 대비 향상
    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)

    # Step 3: Unsharp Masking — 윤곽 강조 (결함 경계 선명화)
    # 가우시안 블러(흐린 버전)를 원본에서 빼서 고주파 성분 강조
    gaussian = cv2.GaussianBlur(img_clahe, (0, 0), 2.0)
    return cv2.addWeighted(img_clahe, 1.5, gaussian, -0.5, 0)
```

이 세 단계는 순서가 중요합니다:
- **BilateralFilter 먼저:** 노이즈를 제거해야 CLAHE가 노이즈를 과도하게 강조하지 않음
- **CLAHE 다음:** 균일한 밝기 기반을 만들어야 Unsharp Masking이 정확히 동작
- **Unsharp Masking 마지막:** 최종 출력에서 결함 경계를 선명하게 강조

### 3.5 Stage 2: EfficientNet-B0 정밀 분류

**`EfficientNet Transfer Learning.py`의 파인튜닝 전략:**

```python
# EfficientNet Transfer Learning.py — 4클래스 분류기 구성
classifier = models.efficientnet_b0(weights='IMAGENET1K_V1')  # ImageNet 사전학습 가중치

# 기존 1000클래스 분류 헤드를 4클래스로 교체
num_features = classifier.classifier[1].in_features
classifier.classifier[1] = nn.Linear(num_features, 4)

# 초기 레이어는 동결, 후반 레이어만 파인튜닝 (선택적 전이학습)
for name, param in classifier.named_parameters():
    if 'features.0' in name or 'features.1' in name:
        param.requires_grad = False
```

**`vision_ai_main.py`에서의 런타임 로드:**

```python
# vision_ai_main.py — EfficientNet 런타임 로드
CLASS_NAMES = ['Defect', 'Normal_C4', 'Normal_Comm', 'Normal_Supply']

classifier = models.efficientnet_b0(weights=None)
classifier.classifier = nn.Linear(
    classifier.classifier.in_features, len(CLASS_NAMES))
classifier.load_state_dict(
    torch.load(EFFNET_WEIGHTS_PATH, map_location=DEVICE))
classifier.eval()

# 추론
with torch.no_grad():
    output = classifier(input_tensor)
    probs = torch.softmax(output, 1)
    conf_score, pred_idx = torch.max(probs, 1)
    conf_percent = conf_score.item() * 100
```

`torch.softmax`는 4개 클래스의 로짓(logit) 값을 합이 1인 확률 분포로 변환합니다.
`torch.max(probs, 1)`은 가장 높은 확률 값과 그 인덱스를 반환합니다.
이 `conf_percent`가 2단 정확도 필터의 입력으로 사용됩니다.

### 3.6 vision_ai_main.py 핵심 메커니즘 심층 분석

**스레드 안전 전역 변수 구조:**

```python
# vision_ai_main.py — 스레드 간 안전한 데이터 공유
lock               = threading.Lock()
global_frame       = None      # MJPEG 스트리밍용 최신 프레임
trigger_flag       = False     # PLC 촬영 신호
is_running         = True      # 시스템 실행 상태
current_status     = "WAITING" # 현재 판정 상태
current_confidence = 0.0       # 최근 신뢰도
current_part_name  = ""        # 현재 부품명
current_order_id   = ""        # 현재 주문 ID
log_queue          = queue.Queue(maxsize=100)  # 터미널 로그 비동기 전송
```

`threading.Lock()`은 `global_frame`을 여러 스레드가 동시에 읽고 쓸 때
데이터 손상을 방지하는 뮤텍스(Mutex)입니다.
`api_server.py`의 `/video_feed`와 `vision_ai_main`의 카메라 루프가
동시에 `global_frame`에 접근하기 때문에 반드시 필요합니다.

**2단 정확도 필터의 실제 구현 로직:**

```python
# vision_ai_main.py — 2단 정확도 필터
MIN_PASS_ACCURACY = 80.0
IGNORE_ACCURACY   = 50.0

if conf_percent < IGNORE_ACCURACY:
    # 50% 미만 → 허상/반사광 노이즈로 판단, 완전 무시
    pass

elif conf_percent < MIN_PASS_ACCURACY:
    # 50~80% → 불확실 케이스: Check! 표시 + 자동 저장
    save_path = f"uncertain/{pred_class}_{timestamp}.jpg"
    save_queue.put((save_path, cropped_img))  # 비동기 저장

else:
    # 80% 이상 → 운영 판정 가능
    if trigger_flag:
        final_result = "FAIL" if pred_class == "Defect" else "PASS"
        with lock:
            current_status     = final_result
            current_confidence = conf_percent
        trigger_flag = False
```

### 3.7 통신 아키텍처 (api_server.py 심층 분석)

**세 가지 통신 패턴의 설계 의도:**

| 엔드포인트 | 패턴 | 설계 의도 |
|-----------|------|-----------|
| `POST /trigger` | Fire-and-Forget | PLC는 신호를 보내고 즉시 다음 작업으로 |
| `GET /result` | Polling (0.5초 주기) | 단순하고 신뢰성 높은 상태 조회 |
| `GET /video_feed` | Server-Sent Events (MJPEG) | 웹 대시보드 실시간 시각화 |

**`/video_feed`의 async 생성기 구조:**

```python
# api_server.py — 비동기 MJPEG 스트리밍
@app.get('/video_feed')
async def video_feed():
    async def generate():
        while vision_ai_main.is_running:
            # Lock으로 보호: vision_ai_main이 frame을 업데이트하는 동안 읽기 방지
            with vision_ai_main.lock:
                if vision_ai_main.global_frame is not None:
                    current_frame = vision_ai_main.global_frame.copy()

            ret, jpeg = cv2.imencode('.jpg', current_frame,
                                     [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + jpeg.tobytes() + b'\r\n')

            await asyncio.sleep(0.05)  # 20fps, 이벤트 루프 양보

    return StreamingResponse(generate(),
        media_type="multipart/x-mixed-replace; boundary=frame")
```

`await asyncio.sleep(0.05)`가 핵심입니다.
이것이 없으면 `while` 루프가 이벤트 루프를 독점해 다른 API 요청을 처리할 수 없게 됩니다.
`failed_attempts/`의 `camera_monitor_yolo.py`가 겪었던 블로킹 문제와 동일한 원리의 해결책입니다.

### 3.8 시행착오에서 최종 아키텍처로의 진화

두 폴더의 관계는 단순한 "실패 → 성공"이 아닙니다.
`failed_attempts/`에서 쌓인 모든 기술 자산이 `two_stage_pipeline/`으로 이전되었습니다.

| failed_attempts/ 자산 | two_stage_pipeline/ 계승 | 변환 내용 |
|----------------------|--------------------------|-----------|
| `Intrinsic Calibration.py` MTX/DIST | `vision_ai_main.py` 하드코딩 | 측정값 → 런타임 상수 |
| `camera_monitor_yolo.py` 카메라 루프 | `vision_ai_main.py` `run_pipeline()` | 동기 → 멀티스레드 |
| `unified_vision_system.py` 스레드 시도 | `api_server.py` + `vision_ai_main.py` 분리 | 단일 파일 → 역할 분리 |
| `API_DB_Check.py` FastAPI 실험 | `api_server.py` Non-blocking 구조 | 블로킹 → Non-blocking |
| 3D 증강 실패 교훈 | `balance_dataset.py` 실제 데이터 증강 | 시뮬레이션 → 실사 기반 |
| V1 `cv2.resize()` 찌그러뜨림 | `letterbox_image()` 비율 유지 | 강제 리사이즈 → Letterbox |

이 표가 이 프로젝트의 본질적인 이야기입니다.
모든 실험은 버려진 것이 아니라 더 나은 형태로 다음 버전에 통합되었습니다.

---

<div align="center">

**이 보고서는 `vision_ai_main.py`, `api_server.py`, `EfficientNet Transfer Learning.py`,  
`balance_dataset.py`, `OBB_Data_Crop.py`, `train_seed_obb.py` 실제 코드를 기반으로 작성되었습니다.**

*GitHub Repository: https://github.com/nicknick111/2026Programming-Study*

</div>

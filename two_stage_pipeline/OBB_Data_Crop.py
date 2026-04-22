import cv2
import numpy as np
import os
import glob
from ultralytics import YOLO
from rembg import remove  # 배경 제거 라이브러리 추가

# ==========================================
# 1. 경로 및 설정
# ==========================================
weights_path = r"E:\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\runs\obb\train\weights\best.pt"
target_images_dir = r"E:\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\Dataset_OBB\test\images"
output_dir = r"E:\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\Extracted_Objects"

# 레터박스 목표 해상도 설정 (정사각형 권장, 예: 640x640)
TARGET_SIZE = (640, 640)

os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 2. 레터박스(Letterbox) 함수 정의
# ==========================================
def apply_letterbox(image, target_size=(640, 640), color=(0, 0, 0, 0)):
    """
    이미지의 원본 비율을 유지하면서 타겟 사이즈에 맞게 조절하고,
    남는 공간을 특정 색상(기본값: 투명)으로 채웁니다.
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # 비율을 유지하기 위한 축소/확대 스케일 계산
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # 이미지 크기 조절 (비율 유지)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 상하좌우에 추가할 여백(패딩) 크기 계산
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left

    # 여백 추가 (cv2.BORDER_CONSTANT 사용, 투명 배경으로 채움)
    letterboxed = cv2.copyMakeBorder(
        resized, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=color
    )
    return letterboxed

# ==========================================
# 3. 모델 로드 및 이미지 처리
# ==========================================
print("모델을 불러오는 중입니다...")
model = YOLO(weights_path)
image_paths = glob.glob(os.path.join(target_images_dir, "*.jpg"))
print(f"총 {len(image_paths)}개의 이미지를 처리합니다.\n")

for img_path in image_paths:
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"이미지를 읽을 수 없습니다: {img_name}")
        continue

    results = model.predict(img, verbose=False)
    
    for result in results:
        if result.obb is not None:
            obbs = result.obb.xyxyxyxy.cpu().numpy() 
            classes = result.obb.cls.cpu().numpy()
            
            for i, (obb, cls) in enumerate(zip(obbs, classes)):
                rect = np.array(obb, dtype="float32")
                (tl, tr, br, bl) = rect
                
                widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
                widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
                maxWidth = max(int(widthA), int(widthB))
                
                heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
                heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
                maxHeight = max(int(heightA), int(heightB))
                
                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]
                ], dtype="float32")
                
                # 1단계: OBB 원근 변환으로 객체만 반듯하게 잘라내기
                M = cv2.getPerspectiveTransform(rect, dst)
                warped_img = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
                
                # 2단계: 배경 제거 (rembg 라이브러리 활용)
                # 이 과정을 거치면 이미지가 BGR 형태에서 투명도를 가진 BGRA(4채널) 형태로 바뀝니다.
                no_bg_img = remove(warped_img)
                
                # 3단계: 레터박스 적용
                # 목표 해상도(기본 640x640)의 정중앙에 객체를 배치하고 남는 공간은 투명하게(0,0,0,0) 채웁니다.
                final_img = apply_letterbox(no_bg_img, target_size=TARGET_SIZE)
                
                # 파일 저장 (투명도를 보존하기 위해 반드시 .png로 저장해야 합니다)
                file_base_name = os.path.splitext(img_name)[0]
                save_filename = f"{file_base_name}_cls{int(cls)}_obj{i}.png"
                save_path = os.path.join(output_dir, save_filename)
                
                cv2.imwrite(save_path, final_img)

print("\n배경 제거 및 레터박스 처리가 완료된 객체 저장이 끝났습니다!")
print(f"저장된 경로: {output_dir}")
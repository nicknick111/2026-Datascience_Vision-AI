import os
import cv2
import math
import numpy as np
from ultralytics import YOLO

# --- [고급 전처리 파이프라인] ---
def apply_advanced_preprocessing(part_img):
    # 1. Bilateral Filter (노이즈 제거)
    filtered = cv2.bilateralFilter(part_img, d=9, sigmaColor=50, sigmaSpace=75)
    # 2. CLAHE (조명/그림자 보정)
    lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    # 3. Unsharp Masking (크랙/적층결 극대화)
    gaussian = cv2.GaussianBlur(img_clahe, (0, 0), 2.0)
    unsharp = cv2.addWeighted(img_clahe, 1.5, gaussian, -0.5, 0)
    return unsharp

# --- [비율 보존 레터박싱] ---
def letterbox_image(img, target_size=224):
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_off, y_off = (target_size - new_w) // 2, (target_size - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas

def main():
    # 1. 방금 학습이 끝난 최고 성능의 OBB 가중치 로드
    model = YOLO(r"E:\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\runs\obb\final_obb_extractor\weights\best.pt")
    
    # 2. Split_Data 폴더 내의 모든 이미지 경로
    base_split_dir = r"E:\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\Dataset_OBB"
    
    # 3. 출력될 증명사진 저장소
    output_dir = r"E:\Robot_Team_Project\EfficientNet_Dataset\Raw_Crops"
    os.makedirs(output_dir, exist_ok=True)
    
    success_cnt = 0
    print("🔄 [Phase 2] 전처리 및 정규화 크롭 추출 시작 (배경 제거 100%)...")
    
    for split_folder in ['train', 'val', 'test']:
        img_dir = os.path.join(base_split_dir, split_folder, 'images')
        if not os.path.exists(img_dir): continue
        
        for filename in os.listdir(img_dir):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')): continue
            
            img_path = os.path.join(img_dir, filename)
            img = cv2.imread(img_path)
            
            # OBB 추론 (conf=0.5로 설정하여 확실한 부품만 검출)
            results = model(img, conf=0.5, verbose=False)
            for r in results:
                if r.obb is None: continue
                
                for box in r.obb.xywhr:
                    cx, cy, w, h, angle_rad = box.cpu().numpy()
                    
                    # 수평 정렬 (Warp Perspective)
                    M = cv2.getRotationMatrix2D((cx, cy), math.degrees(angle_rad), 1.0)
                    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
                    
                    # 타이트 크롭 (Tight Crop)
                    x1, y1 = max(0, int(cx - w / 2)), max(0, int(cy - h / 2))
                    x2, y2 = min(rotated.shape[1], int(cx + w / 2)), min(rotated.shape[0], int(cy + h / 2))
                    cropped = rotated[y1:y2, x1:x2]
                    
                    if cropped.size == 0: continue
                    
                    # [핵심] 순수 부품 영역에 고급 전처리 및 레터박싱 적용
                    preprocessed_crop = apply_advanced_preprocessing(cropped)
                    final_photo = letterbox_image(preprocessed_crop, 224)
                    
                    cv2.imwrite(os.path.join(output_dir, f"crop_{success_cnt:04d}_{filename}"), final_photo)
                    success_cnt += 1

    print(f"✅ 총 {success_cnt}장의 224x224 무결점 증명사진 추출 완료! 저장 위치: {output_dir}")

if __name__ == '__main__':
    main()
import os
import cv2
import random
import numpy as np
from pathlib import Path

def apply_random_augmentation(img):
    """
    7가지 증강 기법 중 하나를 무작위로 선택하여 이미지에 적용합니다.
    """
    aug_type = random.choice(['flip', 'bright', 'dark', 'shift', 'contrast', 'noise', 'rotate'])
    
    if aug_type == 'flip':
        return cv2.flip(img, 1)
        
    elif aug_type == 'bright':
        matrix = np.ones(img.shape, dtype="uint8") * random.randint(20, 50)
        return cv2.add(img, matrix)
        
    elif aug_type == 'dark':
        matrix = np.ones(img.shape, dtype="uint8") * random.randint(20, 50)
        return cv2.subtract(img, matrix)
        
    elif aug_type == 'shift':
        h, w = img.shape[:2]
        tx = random.randint(-20, 20)
        ty = random.randint(-20, 20)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(img, M, (w, h))
        
    elif aug_type == 'contrast':
        alpha = random.uniform(0.7, 1.3)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        
    elif aug_type == 'noise':
        noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
        return cv2.add(img, noise)
        
    elif aug_type == 'rotate':
        h, w = img.shape[:2]
        angle = random.randint(-15, 15)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h))
        
    return img

def balance_dataset():
    target_base_dir = Path(r"E:\Robot_Team_Project\EfficientNet_Dataset\Safe_Split_Dataset\train")
    categories = ["Defect", "Normal_C4", "Normal_Comm", "Normal_Supply"]
    
    # 💡 각 클래스마다 무조건 N장씩 추가 생성합니다.
    additional_aug_count = 1000  # <--- 원하시는 '추가 생성 개수'로 변경하세요!
    
    print(f"🚀 '순수 원본' 데이터만 사용하여 무조건 데이터 증강을 시작합니다! (클래스당 {additional_aug_count}장 추가 생성)\n")

    for category in categories:
        cls_dir = target_base_dir / category
        
        if not cls_dir.exists():
            print(f"⚠️ 경고: {cls_dir.name} 폴더가 존재하지 않습니다. 건너뜁니다.")
            continue
            
        # 해당 폴더에 있는 모든 .jpg 파일을 리스트로 가져옵니다.
        all_images = list(cls_dir.glob("*.jpg"))
        
        # 💡 [핵심 복구 사항] 'aug_'로 시작하는 기존 증강본은 제외하고, 순수 원본만 남깁니다!
        # 이렇게 하면 노이즈가 겹치거나 형태가 과도하게 훼손되는 일을 막을 수 있습니다.
        original_images = [img for img in all_images if not img.name.startswith("aug_")]
        
        if len(original_images) == 0:
            print(f"⚠️ [{category}] 증강할 '순수 원본' 이미지가 단 한 장도 없습니다.")
            continue
            
        print(f"⚙️ [{category}] 증강 작업 중... (순수 원본 {len(original_images)}장만 활용 -> {additional_aug_count}장 추가)")
        
        for i in range(additional_aug_count):
            # 오직 '원본 이미지' 중에서만 무작위로 하나를 뽑습니다.
            img_path = random.choice(original_images)
            img = cv2.imread(str(img_path))
            
            if img is None:
                continue
                
            aug_img = apply_random_augmentation(img)
            
            # 새 파일명 규칙: 덮어쓰기 방지를 위해 랜덤 난수 6자리 부여
            rand_id = random.randint(100000, 999999)
            
            # 파일명 예시: aug_123456_원본이름.jpg
            new_filename = f"aug_{rand_id}_{img_path.name}"
            save_path = cls_dir / new_filename
            
            cv2.imwrite(str(save_path), aug_img)
            
    print("\n🎉 순수 원본 데이터 기반의 안전한 추가 데이터 증강이 완료되었습니다!")

if __name__ == "__main__":
    balance_dataset()
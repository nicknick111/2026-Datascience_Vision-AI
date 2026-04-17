import os
import shutil
import random
from pathlib import Path

def merge_and_split_dataset():
    # 1. 기본 경로 설정 (윈도우 경로)
    # pathlib의 Path를 사용하면 경로 연결과 관리가 훨씬 직관적이고 안전합니다.
    src_base_dir = Path(r"E:\Robot_Team_Project\EfficientNet_Dataset\Raw_Crops")
    dest_base_dir = Path(r"E:\Robot_Team_Project\EfficientNet_Dataset\Safe_Split_Dataset")
    
    # 2. 분류할 클래스(폴더명) 목록
    categories = ["Defect", "Normal_C4", "Normal_Comm", "Normal_Supply"]
    
    # 3. 분할 비율 설정
    train_ratio = 0.7
    val_ratio = 0.2
    # test_ratio는 나머지 0.1이 자동으로 할당됩니다.
    
    print("🚀 원본 이미지 및 LeftPart 이미지 병합/분할 작업을 시작합니다!\n")

    for category in categories:
        # 각 카테고리의 원본 폴더와 LeftPart 폴더 경로 생성
        orig_dir = src_base_dir / category
        left_dir = src_base_dir / f"{category}_LeftPart"
        
        all_images = [] # 원본과 크롭 이미지를 모두 담을 리스트
        
        # 4. 원본 이미지 불러오기
        if orig_dir.exists():
            for img_file in orig_dir.glob("*.*"):
                if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                    # (파일경로, 크롭이미지 여부) 형태로 리스트에 저장합니다.
                    # False는 "원본 이미지"라는 뜻입니다.
                    all_images.append((img_file, False))
        else:
            print(f"⚠️ 경고: 원본 폴더를 찾을 수 없습니다 -> {orig_dir}")
            
        # 5. 크롭(LeftPart) 이미지 불러오기
        if left_dir.exists():
            for img_file in left_dir.glob("*.*"):
                if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                    # True는 "크롭(LeftPart) 이미지"라는 뜻입니다.
                    all_images.append((img_file, True))
        else:
            print(f"⚠️ 경고: LeftPart 폴더를 찾을 수 없습니다 -> {left_dir}")

        total_count = len(all_images)
        if total_count == 0:
            print(f"[{category}] ⚠️ 이미지가 하나도 없습니다. 건너뜁니다.")
            continue

        # 6. 이미지 무작위 섞기 (원본과 LeftPart가 골고루 섞이도록)
        random.seed(42) # 언제 실행해도 똑같이 섞이도록 고정
        random.shuffle(all_images)
        
        # 7. 비율에 따라 리스트를 나눌 기준점(인덱스) 계산
        train_index = int(total_count * train_ratio)
        val_index = int(total_count * (train_ratio + val_ratio))
        
        # 데이터를 3개의 덩어리로 쪼개기
        splits = {
            'train': all_images[:train_index],
            'val': all_images[train_index:val_index],
            'test': all_images[val_index:]
        }
        
        print(f"▶ [{category}] 총 {total_count}장 처리 중... "
              f"(Train: {len(splits['train'])} / Val: {len(splits['val'])} / Test: {len(splits['test'])})")

        # 8. 쪼개진 덩어리들을 각각의 목적지 폴더로 복사
        for split_name, files in splits.items():
            # 예: Safe_Split_Dataset\train\Defect 폴더 생성
            dest_dir = dest_base_dir / split_name / category
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            for file_path, is_left_part in files:
                # [중요] 이름 충돌 방지 로직
                # 만약 LeftPart 이미지라면, 파일명 끝에 "_left"를 붙여줍니다.
                # 예: img_01.jpg -> img_01_left.jpg
                if is_left_part:
                    new_filename = f"{file_path.stem}_left{file_path.suffix}"
                else:
                    new_filename = file_path.name # 원본은 그대로
                
                # 최종 저장될 파일 경로
                dest_file_path = dest_dir / new_filename
                
                # 파일 복사
                shutil.copy2(str(file_path), str(dest_file_path))

    print("\n✅ 모든 데이터의 병합 및 분할 복사 작업이 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    merge_and_split_dataset()
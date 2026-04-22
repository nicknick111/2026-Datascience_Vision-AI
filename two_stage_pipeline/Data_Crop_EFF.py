import cv2
import os
from pathlib import Path

def extract_left_components():
    # 1. 원본 이미지가 있는 폴더 경로 리스트
    # (주의: Windows 경로 문자열 처리를 위해 앞에 'r'을 붙이거나 '\\'로 작성합니다)
    input_paths = [
        r"E:\Robot_Team_Project\EfficientNet_Dataset\Raw_Crops\Defect",
        r"E:\Robot_Team_Project\EfficientNet_Dataset\Raw_Crops\Normal_C4",
        r"E:\Robot_Team_Project\EfficientNet_Dataset\Raw_Crops\Normal_Comm",
        r"E:\Robot_Team_Project\EfficientNet_Dataset\Raw_Crops\Normal_Supply"
    ]

    # 2. 잘라낼 가로 비율 설정 (현재 33%로 설정)
    # 이미지에 따라 이 값을 0.25(25%) ~ 0.4(40%) 등으로 조절하며 최적의 값을 찾으세요.
    crop_ratio = 0.33 

    # 각 폴더별로 순회하며 작업 진행
    for path_str in input_paths:
        input_dir = Path(path_str)
        
        # 3. 원본 보존을 위해 새로운 저장 폴더 생성 (기존 폴더명 뒤에 '_LeftPart' 추가)
        output_dir = Path(str(input_dir) + "_LeftPart")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"작업 시작: {input_dir.name} -> {output_dir.name} 에 저장됩니다.")
        
        # 폴더 내의 이미지 파일(jpg, png 등) 갯수 카운트용
        processed_count = 0

        # 4. 폴더 내의 모든 .jpg 파일 처리 (필요시 .png 등 확장자 추가/변경 가능)
        for img_file in input_dir.glob("*.jpg"):
            # 이미지 읽기
            img = cv2.imread(str(img_file))
            
            # 이미지를 정상적으로 불러오지 못한 경우 건너뜀 (오류 방지)
            if img is None:
                print(f"  경고: {img_file.name} 파일을 읽을 수 없습니다.")
                continue

            # 5. 이미지 크기 정보 가져오기 (세로, 가로, 채널)
            height, width = img.shape[:2]
            
            # 6. 왼쪽 자를 영역의 픽셀 너비 계산
            crop_width = int(width * crop_ratio)
            
            # 7. 이미지 자르기 (NumPy 슬라이싱 활용)
            # img[세로 시작:끝, 가로 시작:끝] -> 세로는 전부 유지, 가로는 처음(0)부터 계산된 너비까지
            cropped_img = img[:, :crop_width]
            
            # 8. 자른 이미지를 새로운 폴더에 원본과 같은 이름으로 저장
            save_path = output_dir / img_file.name
            cv2.imwrite(str(save_path), cropped_img)
            
            processed_count += 1
            
        print(f"완료: 총 {processed_count}개의 이미지가 성공적으로 추출되었습니다.\n")

if __name__ == "__main__":
    extract_left_components()
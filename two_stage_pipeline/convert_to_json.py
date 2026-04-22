import os
import json
import cv2

def main():
    # 1. 이미지와 txt 파일이 함께 들어있는 폴더 경로 (장원영 님의 실제 경로로 수정하세요)
    folder_path = r"E:\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\Dataset_OBB\1차 파일\train\images"
    
    # 2. 클래스 이름 설정 (0번 클래스)
    class_names = ["Part"]

    # 변환된 파일 개수 카운트
    success_count = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt") and filename != "classes.txt":
            txt_path = os.path.join(folder_path, filename)
            # txt 파일 이름과 동일한 jpg 이미지 찾기
            img_path = os.path.join(folder_path, filename.replace(".txt", ".jpg"))

            if not os.path.exists(img_path):
                continue

            # 3. 이미지 해상도 읽기 (정규화된 좌표를 픽셀 좌표로 되돌리기 위해 필수)
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w, _ = img.shape

            shapes = []
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    # OBB 포맷(클래스 1개 + 좌표 8개 = 총 9개)인 경우만 처리
                    if len(parts) == 9: 
                        class_id = int(parts[0])
                        label = class_names[class_id]

                        # 정규화된 좌표(0~1)를 실제 이미지의 픽셀 좌표로 변환
                        points = [
                            [float(parts[1]) * w, float(parts[2]) * h],
                            [float(parts[3]) * w, float(parts[4]) * h],
                            [float(parts[5]) * w, float(parts[6]) * h],
                            [float(parts[7]) * w, float(parts[8]) * h]
                        ]

                        shapes.append({
                            "label": label,
                            "points": points,
                            "group_id": None,
                            "shape_type": "polygon",  # AnyLabeling이 폴리곤으로 인식하게 함
                            "flags": {}
                        })

            # 4. AnyLabeling 호환 JSON 구조 생성
            json_data = {
                "version": "0.3.3",
                "flags": {},
                "shapes": shapes,
                "imagePath": os.path.basename(img_path),
                "imageData": None,
                "imageHeight": h,
                "imageWidth": w
            }

            # 5. 동일한 이름의 .json 파일로 저장
            json_path = os.path.join(folder_path, filename.replace(".txt", ".json"))
            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump(json_data, jf, ensure_ascii=False, indent=2)
            
            success_count += 1

    print(f"✅ 총 {success_count}개의 파일이 AnyLabeling용 JSON으로 완벽하게 변환되었습니다!")

if __name__ == '__main__':
    main()
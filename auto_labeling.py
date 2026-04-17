import os
from ultralytics import YOLO

def main():
    # 1. 방금 학습 완료된 Seed 모델 가중치 로드
    model_path = r"E:\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\runs\obb\seed_obb_v1\weights\best.pt"
    model = YOLO(model_path)

    # 2. 남은 736장 이미지 폴더 및 텍스트 파일 저장 폴더 설정
    input_images_dir = r"E:\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\unlabeled_images"
    output_labels_dir = r"E:\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\auto_labels"
    
    os.makedirs(output_labels_dir, exist_ok=True)

    print("🤖 AI가 736장의 4점 폴리곤 자동 라벨링을 시작합니다...")
    
    # 3. 추론 실행 (conf=0.5: 확신도가 50% 이상인 것만 그리기)
    results = model.predict(source=input_images_dir, conf=0.5, save=False)

    # 4. 추론 결과(OBB 좌표)를 YOLO 텍스트(.txt) 포맷으로 변환하여 저장
    success_count = 0
    for result in results:
        img_name = os.path.basename(result.path)
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(output_labels_dir, txt_name)

        with open(txt_path, 'w') as f:
            if result.obb is not None:
                # result.obb.xyxyxyxyn: 0~1 사이로 정규화된 4개 꼭짓점 좌표 (x1 y1 x2 y2 x3 y3 x4 y4)
                for box, cls in zip(result.obb.xyxyxyxyn, result.obb.cls):
                    pts = box.cpu().numpy().flatten()
                    # 8개의 좌표를 소수점 6자리까지 문자열로 띄어쓰기 변환
                    pts_str = " ".join([f"{pt:.6f}" for pt in pts])
                    f.write(f"{int(cls)} {pts_str}\n")
                    success_count += 1
                    
    print(f"✅ 자동 라벨링 완료! 총 {success_count}개의 객체가 '{output_labels_dir}' 폴더에 .txt로 저장되었습니다.")

if __name__ == '__main__':
    main()
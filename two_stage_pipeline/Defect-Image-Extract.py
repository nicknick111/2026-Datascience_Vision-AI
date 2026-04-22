import cv2
import os

# =====================================================================
# [1] 고급 전처리 알고리즘 (정상품과 동일한 조건 유지 필수)
# =====================================================================
def preprocess_frame(frame):
    roi_frame = frame.copy() 
    blurred = cv2.bilateralFilter(roi_frame, d=9, sigmaColor=50, sigmaSpace=75)
    
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    gaussian = cv2.GaussianBlur(enhanced_bgr, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced_bgr, 1.2, gaussian, -0.2, 0)
    
    return sharpened

# =====================================================================
# [2] OBB 라벨링을 위한 불량품 이미지 통합 추출 로직
# =====================================================================
def main():
    # 🎯 정상품과 동일한 폴더 지정! (여기에 모두 모아서 한 번에 Roboflow 업로드)
    target_dir = r"E:\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\OBB_Raw_Data"
    os.makedirs(target_dir, exist_ok=True)
    
    # 영상 경로 설정 (BackGround.mp4가 불량품을 의미하는 것으로 보임)
    video_paths = [
        {"path": "불량품_컨베이어 이동.mp4", "prefix": "Defect_v2"}
    ]
    
    frame_interval = 5 # 불량품은 데이터 확보를 위해 5프레임당 1장 추출
    
    print("🚀 불량 부품 영상 전처리 및 라벨링 폴더로 추출 시작...")
    
    total_saved = 0
    for video in video_paths:
        if not os.path.exists(video['path']):
            print(f"⚠️ 경고: {video['path']} 파일을 찾을 수 없습니다.")
            continue
            
        cap = cv2.VideoCapture(video['path'])
        count = 0
        saved_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if count % frame_interval == 0:
                processed = preprocess_frame(frame)
                # target_dir에 모든 이미지 통합 저장
                save_path = os.path.join(target_dir, f"{video['prefix']}_{saved_count:04d}.jpg")
                cv2.imwrite(save_path, processed)
                saved_count += 1
                total_saved += 1
                
            count += 1
            
        cap.release()
        print(f"✅ {video['path']} -> {saved_count}장 추출 완료")

    print(f"\n🎉 불량품 총 {total_saved}장 추출 완료! ({target_dir})")
    print("👉 이제 OBB_Raw_Data 폴더의 모든 이미지를 Roboflow에 업로드하여 라벨링을 시작하세요.")

if __name__ == "__main__":
    main()
import cv2
import os

# =====================================================================
# [1] 고급 전처리 알고리즘 (유사 객체 특징 강화 튜닝)
# =====================================================================
def preprocess_frame(frame):
    roi_frame = frame.copy() 
    
    # 1. 노이즈 제거 (Bilateral Filter)
    blurred = cv2.bilateralFilter(roi_frame, d=9, sigmaColor=50, sigmaSpace=75)
    
    # 2. CLAHE (적응형 히스토그램 평활화)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # 3. 언샤프 마스킹 (경계선 선명하게)
    gaussian = cv2.GaussianBlur(enhanced_bgr, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced_bgr, 1.2, gaussian, -0.2, 0)
    
    return sharpened

# =====================================================================
# [2] OBB 라벨링을 위한 정상품 이미지 통합 추출 로직
# =====================================================================
def main():
    # 🎯 모든 이미지가 모일 통합 라벨링 대기 폴더
    target_dir = r"E:\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\OBB_Raw_Data"
    os.makedirs(target_dir, exist_ok=True)
    
    # 영상 경로 설정
    video_paths = [
        {"path": "정상품_C4_컨베이어 이동.mp4", "prefix": "Normal_C4"},
        {"path": "정상품_communication_컨베이어 이동.mp4", "prefix": "Normal_Comm"},
        {"path": "정상품_Supply_컨베이어 이동.mp4", "prefix": "Normal_Supply"}
    ]
    
    frame_interval = 10 # 10프레임당 1장 추출
    
    print("🚀 정상 부품 영상 전처리 및 라벨링 폴더로 추출 시작...")
    
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

    print(f"\n🎉 정상품 총 {total_saved}장 추출 완료! ({target_dir})")

if __name__ == "__main__":
    main()
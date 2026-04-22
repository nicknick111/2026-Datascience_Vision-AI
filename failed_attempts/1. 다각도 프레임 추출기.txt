import cv2
import os
import numpy as np

def variance_of_laplacian(image):
    """이미지의 흐릿함(Blur) 정도를 계산합니다. (값이 낮을수록 흐릿함)"""
    return cv2.Laplacian(image, cv2.CV_64F).var()

def extract_smart_frames(video_path, output_folder, prefix, step_frames=15, blur_threshold=100.0):
    """
    [3D 다각도 인식 실패 예방 1단계]
    영상의 모든 프레임을 뽑으면 똑같은 각도만 수천 장이 되어 AI가 바보가 됩니다(과적합).
    따라서 일정 간격(step_frames)으로 건너뛰며 '각도가 변한' 사진만 추출하고,
    초점이 나가서 흐릿한 사진은 자동으로 버리는 스마트 추출기입니다.
    """
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ 오류: 영상을 열 수 없습니다 -> {video_path}")
        return

    frame_count = 0
    saved_count = 0

    print(f"🎬 영상 추출 시작: {video_path}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 일정 프레임(예: 15프레임 = 0.5초)마다 1장씩만 검사
        if frame_count % step_frames == 0:
            # 2. 이미지가 너무 흔들려서 흐릿한지(Motion Blur) 검사
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fm = variance_of_laplacian(gray)
            
            # 3. 선명한 사진만 저장
            if fm > blur_threshold:
                # YOLOv8 학습에 유리하게 미리 640x640 크기로 조정 (비율 무시 단순 조정 예시)
                # 실제 현장에서는 좌우 여백을 검은색으로 채우는 Letterbox 기법을 권장합니다.
                resized_frame = cv2.resize(frame, (640, 640))
                
                # 파일명 예: RightWing_00001.jpg
                filename = f"{prefix}_{saved_count:05d}.jpg"
                save_path = os.path.join(output_folder, filename)
                
                # 한글 경로 에러 방지를 위한 numpy 기반 저장 방식
                result, encoded_img = cv2.imencode('.jpg', resized_frame)
                if result:
                    with open(save_path, mode='w+b') as f:
                        encoded_img.tofile(f)
                
                saved_count += 1
            else:
                pass # 흐릿한 사진은 AI를 망치므로 버림

        frame_count += 1

    cap.release()
    print(f"✅ {prefix} 완료! 총 {saved_count}장의 다각도 '선명한' 대표 이미지가 추출되었습니다.\n")

if __name__ == "__main__":
    # 방금 업로드해주신 영상 파일들의 목록입니다.
    video_list = [
        {"path": "20260323_Right_wing.mp4", "prefix": "RightWing"},
        {"path": "Uav_Fuselage_part_1.mp4", "prefix": "Fuselage_1"},
        {"path": "ISR_System_part_1.mp4", "prefix": "ISR_System"},
        {"path": "Internal_Assembly_of_Model_UAV_Fuselage_Part_3_1.mp4", "prefix": "Fuselage_3_1"}
    ]
    
    # 추출된 이미지가 저장될 폴더
    output_dir = "./dataset/raw_images"
    
    for video in video_list:
        if os.path.exists(video["path"]):
            # step_frames=15 : 30fps 영상 기준 1초에 2장씩 추출하여 각도 변화 확보
            extract_smart_frames(video["path"], output_dir, video["prefix"], step_frames=15)
        else:
            print(f"⚠️ 경고: {video['path']} 파일이 같은 폴더에 없습니다.")
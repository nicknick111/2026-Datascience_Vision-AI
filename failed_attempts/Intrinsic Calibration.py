import cv2
import numpy as np
import os

# =====================================================================
# [1] 캘리브레이션 환경 설정
# =====================================================================
# 체스보드 내부 교차점 개수 (가로 코너 수, 세로 코너 수)
CHESSBOARD_SIZE = (6, 9) 

# 체스보드 1칸의 실제 길이 (단위: 미터) - 필요시 실측 후 수정
SQUARE_SIZE = 0.025 

# 알려주신 영상 절대 경로 적용 (r을 붙여 Windows 경로 에러 방지)
VIDEO_PATH = r"E:\Robot_Team_Project\Python_Files(AI Programming)\TestSet02\calib_images\inspection_calibration.mp4"

# 프레임 추출 간격 (15프레임 = 약 0.5초마다 1장씩 검사)
FRAME_INTERVAL = 15  

# =====================================================================
# [2] 데이터 준비 및 영상 처리
# =====================================================================
def run_video_calibration():
    print(f"🚀 동영상 데이터 기반 Intrinsic Calibration을 시작합니다...")
    print(f"📂 타겟 영상: {VIDEO_PATH}")
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ 오류: 동영상을 찾을 수 없거나 열 수 없습니다. 경로를 다시 확인해주세요.")
        return

    # 3D 실제 세계 좌표 준비
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = [] # 3D 실제 좌표
    imgpoints = [] # 2D 픽셀 좌표

    frame_count = 0
    valid_images = 0
    image_shape = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print("영상을 재생하며 체스보드를 스캔합니다. 잠시만 기다려주세요...\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break # 영상 끝

        # 지정된 프레임 간격마다 코너 찾기 시도
        if frame_count % FRAME_INTERVAL == 0:
            # 해상도를 실전 환경(640x480)에 맞춤
            if frame.shape[1] != 640 or frame.shape[0] != 480:
                frame = cv2.resize(frame, (640, 480))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if image_shape is None:
                image_shape = gray.shape[::-1] 
            
            # 체스보드 코너 찾기
            ret_corner, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
            
            if ret_corner:
                objpoints.append(objp)
                
                # 코너 정밀 보정
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                valid_images += 1
                
                # 시각화 (성공한 프레임만 화면에 잠깐 보여줌)
                cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners2, ret_corner)
                cv2.imshow('Scanning Video...', frame)
                cv2.waitKey(100)
        
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 스캔 완료! 총 추출된 유효 프레임: {valid_images} 장")

    if valid_images < 10:
        print("❌ 경고: 코너를 찾은 프레임이 10장 미만입니다. 동영상을 다시 녹화해주세요.")
        print("   (팁: 영상을 천천히 움직이고, 체스보드 여백을 2~3cm 주고, 전처리 필터를 끈 상태로 녹화하세요.)")
        return

    # =====================================================================
    # [3] 카메라 매트릭스 도출 및 데이터 저장
    # =====================================================================
    print("⚙️ 카메라 내부 파라미터(K) 및 왜곡 계수(D) 계산 중...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)
    
    print("\n✅ [결과 1] Camera Matrix (Intrinsic K):")
    print(np.round(mtx, 4))
    print("\n✅ [결과 2] Distortion Coefficients:")
    print(np.round(dist, 4))
    
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
        
    print(f"\n🎯 전체 평균 오차율 (Reprojection Error): {mean_error/len(objpoints):.4f} 픽셀 (※ 0.5 이하 권장)")

    # 저장될 경로: 파이썬 실행 위치 기준 (통합 코드와 같은 폴더에 생성됨)
    save_filename = 'camera_calib_data.npy'
    np.save(save_filename, {'mtx': mtx, 'dist': dist})
    print(f"\n🎉 완료! '{save_filename}' 파일이 생성되었습니다.")

if __name__ == "__main__":
    run_video_calibration()
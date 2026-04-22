from ultralytics import YOLO

# 1. 학습된 best.pt 모델 불러오기
model_path = "E:/Robot_Team_Project/Python_Files(AI Programming)/TestSet02/best.pt"
model = YOLO(model_path)

# 2. X-AnyLabeling 호환을 위해 ONNX 포맷으로 변환 (동일한 폴더에 best.onnx가 생성됩니다)
print("ONNX 변환을 시작합니다...")
model.export(format='onnx', imgsz=640)
print("✅ 변환 완료! best.onnx 파일이 생성되었습니다.")
from ultralytics import YOLO

def train_rep_model():
    model = YOLO("yolo26n.pt")   # 또는 yolo11n.pt 등
    model.train(
        data="dataset_rep.yaml",
        epochs=20,
        imgsz=640,
        batch=16,
        device="cpu",
        project="runs_rep",
        name="rep_train"
    )

if __name__ == "__main__":
    train_rep_model()

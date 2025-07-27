import os
from ultralytics import YOLO

def train_model(model_path, data_yaml, device='cuda'):
    """
    Trains a YOLO model on a given dataset.
    """
    model = YOLO(model_path)
    model.train(
        data=data_yaml,
        batch=19,
        imgsz=640,
        patience=0,      # Enable early stopping with some patience
        epochs=350,
        device=device,
        half=True,
        workers=0,
        optimizer='auto',
        name="model_train",
        project="CrossArmTopCleatModel"
    )
    print("✅ Training complete.")

    val_metrics = model.val(split='val')
    test_metrics = model.val(split='test')

    print("📊 Validation metrics:", val_metrics)
    print("📊 Test metrics:", test_metrics)

# === CONFIG ===
dataset_dir = "/home/line_quality/line_quality/Sampath/Models/CrossArmsModel/CrossArmsTopCleatDefects_Dataset/"
data_yaml = os.path.join(dataset_dir, "data.yaml")
train_model('yolo12m.pt', data_yaml)
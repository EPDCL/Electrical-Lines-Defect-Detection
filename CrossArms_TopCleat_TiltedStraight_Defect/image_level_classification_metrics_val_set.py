import os
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

model_path = 'CrossArmTopCleatModel/model_train2/weights/best.pt'  # Path to best trained model
split = 'test'
image_dir = f'/home/line_quality/line_quality/Sampath/Models/CrossArmsModel/CrossArmsTopCleatDefects_Dataset/{split}/images'
gt_path = f'/home/line_quality/line_quality/Sampath/Models/CrossArmsModel/CrossArmsTopCleatDefects_Dataset/{split}/labels'
output_dir = '/home/line_quality/line_quality/Sampath/Models/CrossArmsModel/Outputs/'
project_name = f'predict_{split}'
pred_path = os.path.join(output_dir, project_name, 'labels')
confidence_threshold = 0.35

# class indices in dictionary key:value pair.
class_names = {
    0: 'CrossArm_Straight',
    1: 'CrossArm_Tilted',
    2: 'TopCleat_Straight',
    3: 'TopCleat_Tilted',
    4: 'V-CrossArm_Straight',
    5: 'V-CrossArm_Tilted'
}

# loading trained yolo model
model = YOLO(model_path)
os.makedirs(output_dir, exist_ok=True)

# inference
results = model.predict(
    source=image_dir,
    save=True,
    save_txt=True,
    save_conf=True,
    project=output_dir,
    name=project_name,
    conf=confidence_threshold
)

print(f"Inference complete for {split}. Predictions saved to: {output_dir}/{project_name}")

# Loading labels for GT and preds
gt_files = [f for f in os.listdir(gt_path) if f.endswith(".txt")]
pred_files = [f for f in os.listdir(pred_path) if f.endswith(".txt")]
# Initializing dictionary for each class with loop
results_data = {
    class_id: {"gt": [], "pred": [], "fp_files": [], "fn_files": []}
    for class_id in class_names.keys()
}

# evaluating
for file in gt_files:
    gt_file = os.path.join(gt_path, file)
    pred_file = os.path.join(pred_path, file)

    with open(gt_file, "r") as f:
        gt_classes = set(line.strip().split()[0] for line in f.readlines())

    pred_classes = set()
    if os.path.exists(pred_file):
        with open(pred_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 6:
                    class_id, conf = parts[0], float(parts[5])
                    if conf >= confidence_threshold:
                        pred_classes.add(class_id)

    for class_id in class_names.keys():
        class_str = str(class_id)
        gt_present = 1 if class_str in gt_classes else 0
        pred_present = 1 if class_str in pred_classes else 0

        results_data[class_id]["gt"].append(gt_present)
        results_data[class_id]["pred"].append(pred_present)

        if gt_present != pred_present:
            (results_data[class_id]["fp_files"] if pred_present else results_data[class_id]["fn_files"]).append(file)


extra_preds = set(pred_files) - set(gt_files)
for file in extra_preds:
    pred_file = os.path.join(pred_path, file)
    pred_classes = set()
    with open(pred_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                class_id, conf = parts[0], float(parts[5])
                if conf >= confidence_threshold:
                    pred_classes.add(class_id)

    for class_id in class_names.keys():
        class_str = str(class_id)
        gt_present = 0
        pred_present = 1 if class_str in pred_classes else 0

        results_data[class_id]["gt"].append(gt_present)
        results_data[class_id]["pred"].append(pred_present)

        if pred_present:
            results_data[class_id]["fp_files"].append(file)


# sklearn.metrics report
for class_id, class_name in class_names.items():
    y_true = results_data[class_id]["gt"]
    y_pred = results_data[class_id]["pred"]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nImage-Level Classification Results ({class_name} - {split.upper()} Set):")
    print(f"Accuracy:  {acc:.2%}")
    print(f"Precision: {prec:.2%}")
    print(f"Recall:    {rec:.2%}")
    print(f"F1 Score:  {f1:.2%}")
    print("Confusion Matrix [TN, FP, FN, TP]:\n", cm)
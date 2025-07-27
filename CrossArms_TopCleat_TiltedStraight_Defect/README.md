#  Electric Pole Defect Detection — CrossArms and TopCleat (Straight / Tilted) 

This repository contains training and evaluation code for detecting **CrossArms and TopCleat (Straight / Tilted) ** using **YOLOv12** object detection. 

This repo also includes image-level classification accuracy evaluation scripts based on YOLO predictions.

---

##  Dataset

Dataset hosted on Hugging Face:  
👉 [Object Detection Dataset](https://huggingface.co/datasets/EPDCL/Electrical-Lines-Defect-Detection/tree/main/CrossArmsTopCleatDefects_Dataset)

- 1350 total images from 3 districts in Andhra Pradesh
- Format: YOLOv12-style `.jpg` images and `.txt` annotations
- Labels:
  - `0`: CrossArm\_Straight
  - `1`: CrossArm\_Tilted
  - `2`: TopCleat\_Straight
  - `3`: TopCleat\_Tilted
  - `4`: V-CrossArm\_Straight
  - `5`: V-CrossArm\_Tilted

- Splits:
  - Train: 946 images (70%)
  - Val: 202 images   (15%)
  - Test: 202 images  (15%)

---

## 📁 GitHub Repository Structure

```
Electrical-Lines-Defect-Detection/
└── CrossArmsTopCleatDefects_Dataset/
  ├── train_model.py # YOLOv12 training script
  ├── image_level_classification_metrics_val_set.py  # Inference + metrics on val set
  ├── image_level_classification_metrics_test_set.py # Inference + metrics on test set
  ├── object_detection_dataset_bboxes.json
  ├── README.md (this file)
  └── assets/
```

---

## 📥 Setup: Clone the Dataset
To run training and evaluation, you must first clone the dataset locally and place it in the expected directory:
```bash
# Step 1: Install Git LFS if not already installed
git lfs install

# Step 2: Clone the dataset repo
git clone https://huggingface.co/datasets/EPDCL/Electrical-Lines-Defect-Detection

# Step 3: Move or rename it to match expected path
mv Electrical-Lines-Defect-Detection /path/to/dataset/Electrical-Lines-Defect-Detection
```
Ensure the folder contains the following structure:
```bash
../CrossArmsTopCleatDefects_Dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── data.yaml
└── object_detection_dataset_bboxes.json
```
You may edit the path in ```train_model.py```, ```image_level_classification_metrics_val_set.py ```, and ```image_level_classification_metrics_test_set.py ``` if your local directory differs.

## Training (Use the Jupyter notebook with the pipeline ready / the script)

```bash
python train_model.py
```
Training by default uses:
- Base model: yolo12m.pt
- Batch size: 19
- Image size: 640x640
- Epochs: 350
- Optimizer: auto
- Device: cuda

  ## 🔍Inference & Evaluation
  ###  Run on validation set:
  ```bash
  python image_level_classification_metrics_val_set.py 
  ```
  ###  Run on test set:
  ```bash
  python image_level_classification_metrics_test_set.py 
  ```

## 📊 YOLOv12 Object Detection Performance
#### Hardware Used for training model: NVIDIA GeForce RTX 4070 Ti SUPER on [JOHNAIC](https://von-neumann.ai/index.html)
###  Validation Set
| Class                | Precision | Recall    | mAP\@0.5  | mAP\@0.5:0.95 |
| -------------------- | --------- | --------- | --------- | ------------- |
| CrossArm\_Straight   | 0.917     | 0.927     | 0.895     | 0.775         |
| CrossArm\_Tilted     | 0.900     | 0.861     | 0.894     | 0.780         |
| TopCleat\_Straight   | 0.980     | 0.980     | 0.918     | 0.764         |
| TopCleat\_Tilted     | 0.948     | 0.937     | 0.919     | 0.774         |
| V-CrossArm\_Straight | 0.943     | 0.967     | 0.953     | 0.944         |
| V-CrossArm\_Tilted   | 0.894     | 0.918     | 0.940     | 0.949         |
| **Overall**          | **0.918** | **0.927** | **0.905** | **0.831**     |

##### Speed: 2.8ms/inference, 0.2ms/postprocess per image
###  Test Set
| Class                | Precision | Recall    | mAP\@0.5  | mAP\@0.5:0.95 |
| -------------------- | --------- | --------- | --------- | ------------- |
| CrossArm\_Straight   | 0.917     | 0.957     | 0.953     | 0.811         |
| CrossArm\_Tilted     | 0.963     | 0.940     | 0.968     | 0.837         |
| TopCleat\_Straight   | 0.870     | 0.870     | 0.808     | 0.600         |
| TopCleat\_Tilted     | 0.866     | 0.800     | 0.794     | 0.570         |
| V-CrossArm\_Straight | 0.950     | 0.983     | 0.978     | 0.952         |
| V-CrossArm\_Tilted   | 0.894     | 0.917     | 0.955     | 0.914         |
| **Overall**          | **0.910** | **0.911** | **0.909** | **0.781**     |

##### Speed: 2.7ms/inference, 0.2ms/postprocess per image

## 📊 Image-Level Classification Performance

<p align="center">
  <img src="to be replaced with graph image link" width="600"/>
</p>

#### Evaluated using YOLO prediction outputs aggregated per image.
### Validation Set
| Class                | Accuracy | Precision | Recall | F1 Score |
| -------------------- | -------- | --------- | ------ | -------- |
| CrossArm\_Straight   | 94.06%   | 91.49%    | 95.56% | 93.48%   |
| CrossArm\_Tilted     | 94.06%   | 78.57%    | 91.67% | 84.62%   |
| TopCleat\_Straight   | 93.56%   | 93.41%    | 92.39% | 92.90%   |
| TopCleat\_Tilted     | 94.55%   | 81.48%    | 78.57% | 80.00%   |
| V-CrossArm\_Straight | 96.04%   | 91.14%    | 98.63% | 94.74%   |
| V-CrossArm\_Tilted   | 96.53%   | 91.84%    | 93.75% | 92.78%   |


### Test Set
| Class                | Accuracy | Precision | Recall  | F1 Score |
| -------------------- | -------- | --------- | ------- | -------- |
| CrossArm\_Straight   | 95.54%   | 91.09%    | 100.00% | 95.34%   |
| CrossArm\_Tilted     | 95.54%   | 86.05%    | 92.50%  | 89.16%   |
| TopCleat\_Straight   | 90.59%   | 86.41%    | 94.68%  | 90.36%   |
| TopCleat\_Tilted     | 93.07%   | 81.08%    | 81.08%  | 81.08%   |
| V-CrossArm\_Straight | 96.04%   | 92.71%    | 98.89%  | 95.70%   |
| V-CrossArm\_Tilted   | 96.53%   | 88.24%    | 97.83%  | 92.78%   |


## 📄 License

- **Code:** MIT License  
- **Dataset:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
  Available at: [Hugging Face Dataset](https://huggingface.co/datasets/EPDCL/Electrical-Lines-Defect-Detection/tree/main/CrossArmsTopCleatDefects_Dataset)

## Sample predictions
<p align="center">
  <img src="to be replaced with predictions" width="600"/>
</p>

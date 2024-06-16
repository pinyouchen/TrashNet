import torch
import imgaug.augmenters as iaa
import imageio

print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs available:", torch.cuda.device_count())

# Step 3: 設定數據集和目錄結構
import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

# 定義路徑
data_dir = 'D:/exam/TrashNet'
output_dir = 'D:/exam/TrashNet_yolo'

# 創建 YOLOv5 資料夾結構
os.makedirs(os.path.join(output_dir, 'images/train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images/val'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'images/test'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels/train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels/val'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'labels/test'), exist_ok=True)

# 創建 data.yaml 文件
def create_data_yaml():
    yaml_content = f"""
train: {output_dir}/images/train
val: {output_dir}/images/val
test: {output_dir}/images/test

nc: 6  # number of classes
names: ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']  # class names
"""
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content)

create_data_yaml()

# 資料增強
augmenters = iaa.Sequential([
    iaa.Fliplr(0.5),  
    iaa.Crop(percent=(0, 0.1)),
    iaa.LinearContrast((0.75, 1.5)),
    iaa.Multiply((0.8, 1.2)),
    iaa.Affine(
        rotate=(-20, 20),
        shear=(-10, 10)
    )
])

# 複製圖片並生成標籤文件
def prepare_dataset():
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    for class_name in classes:
        img_files = [f for f in os.listdir(os.path.join(data_dir, class_name)) if f.endswith('.jpg')]
        train_files, test_files = train_test_split(img_files, test_size=0.2, random_state=42)
        val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)
        
        for phase, files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
            for img_file in files:
                src_img_path = os.path.join(data_dir, class_name, img_file)
                dst_img_path = os.path.join(output_dir, 'images', phase, img_file)
                shutil.copyfile(src_img_path, dst_img_path)
                
                label_file = img_file.replace('.jpg', '.txt')
                with open(os.path.join(output_dir, 'labels', phase, label_file), 'w') as f:
                    f.write(f"{classes.index(class_name)} 0.5 0.5 1.0 1.0\n")

prepare_dataset()

# Step 4: 訓練 YOLOv5 模型
import subprocess

def train_yolov5():
    data_yaml_path = os.path.join(output_dir, 'data.yaml')
    weights_path = 'yolov5s.pt'  # 假設 yolov5s.pt 位於當前目錄或已安裝到環境中
    img_size = 640
    batch_size = 16
    epochs = 50
    patience = 5
    project = output_dir
    name = 'exp'
    
    subprocess.run([
        'python', 'train.py', 
        '--data', data_yaml_path, 
        '--epochs', str(epochs), 
        '--batch-size', str(batch_size), 
        '--imgsz', str(img_size), 
        '--weights', weights_path, 
        '--project', project, 
        '--name', name, 
        '--patience', str(patience)
    ])

# 切換到 YOLOv5 目錄
os.chdir('D:/exam/YOLOv5')  # 更改為你的 YOLOv5 目錄

train_yolov5()
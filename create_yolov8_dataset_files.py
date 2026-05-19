import os
import random

def create_yolov8_dataset_files():
    # 数据集路径
    data_dir = "data/coco"
    images_dir = os.path.join(data_dir, "images")
    
    # 获取所有图像文件
    image_files = []
    for file in os.listdir(images_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_files.append(file)
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 随机打乱
    random.seed(42)
    random.shuffle(image_files)
    
    # 划分比例
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    # 划分数据集
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    print(f"训练集: {len(train_files)} 张图像")
    print(f"验证集: {len(val_files)} 张图像")
    print(f"测试集: {len(test_files)} 张图像")
    
    # 创建图像路径列表文件
    def write_list(file_list, filename):
        with open(os.path.join(data_dir, filename), 'w', encoding='utf-8') as f:
            for img_file in file_list:
                # YOLOv8期望相对路径（相对于数据集根目录）
                img_path = f"images/{img_file}"
                f.write(f"{img_path}\n")
    
    write_list(train_files, "train.txt")
    write_list(val_files, "val.txt")
    write_list(test_files, "test.txt")
    
    print("已创建 train.txt, val.txt, test.txt 文件")
    
    # 更新YAML文件
    yaml_content = f"""# COU水下目标检测数据集
# 24类人造物

path: {data_dir}/  # 数据集根目录
train: train.txt  # 训练图像列表
val: val.txt      # 验证图像列表
test: test.txt    # 测试图像列表

# 类别数量
nc: 24

# 类别名称
names:
  0: plastic_bottle
  1: plastic_bag
  2: fishing_net
  3: rope
  4: can
  5: glass_bottle
  6: tire
  7: metal_scrap
  8: wood
  9: cloth
  10: diver
  11: diving_mask
  12: diving_fins
  13: oxygen_tank
  14: underwater_camera
  15: auv
  16: rov
  17: underwater_drone
  18: sonar
  19: underwater_sensor
  20: ship_wreck
  21: anchor
  22: propeller
  23: underwater_structure
"""
    
    with open(os.path.join(data_dir, "coco.yaml"), 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print("已更新 coco.yaml 文件")
    
    # 显示一些示例
    print("\n示例图像路径:")
    print(f"训练集示例: images/{train_files[0] if train_files else 'N/A'}")
    print(f"验证集示例: images/{val_files[0] if val_files else 'N/A'}")
    print(f"测试集示例: images/{test_files[0] if test_files else 'N/A'}")

if __name__ == "__main__":
    create_yolov8_dataset_files()

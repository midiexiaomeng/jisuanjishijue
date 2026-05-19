import os
import yaml

# 检查数据集结构
data_yaml_path = 'data/YOLO/dataset.yaml'
with open(data_yaml_path, 'r', encoding='utf-8') as f:
    data_config = yaml.safe_load(f)

print('数据集配置:')
train_set = data_config.get('train', 'N/A')
val_set = data_config.get('val', 'N/A')
test_set = data_config.get('test', 'N/A')
nc = data_config.get('nc', 'N/A')
names = data_config.get('names', 'N/A')

print(f'训练集: {train_set}')
print(f'验证集: {val_set}')
print(f'测试集: {test_set}')
print(f'类别数: {nc}')
print(f'类别名称: {names}')

# 检查训练图像目录
base_dir = os.path.dirname(data_yaml_path)
train_dir = os.path.join(base_dir, train_set) if train_set != 'N/A' else ''
print(f'\n训练图像目录: {train_dir}')
if train_dir and os.path.exists(train_dir):
    files = os.listdir(train_dir)
    print(f'训练图像数量: {len(files)}')
    print(f'前5个文件: {files[:5]}')
else:
    print('训练图像目录不存在')

# 检查训练标签目录
train_label_dir = os.path.join(base_dir, 'labels', 'train')
print(f'\n训练标签目录: {train_label_dir}')
if os.path.exists(train_label_dir):
    files = os.listdir(train_label_dir)
    print(f'训练标签数量: {len(files)}')
    print(f'前5个文件: {files[:5]}')
    
    # 检查一个标签文件的内容
    if files:
        label_path = os.path.join(train_label_dir, files[0])
        print(f'\n检查标签文件: {label_path}')
        with open(label_path, 'r') as f:
            content = f.read()
            if content:
                print(f'内容: {content[:100]}')
            else:
                print('内容: 空文件')
else:
    print('训练标签目录不存在')

# 检查是否有对应的图像文件
if os.path.exists(train_label_dir) and files:
    # 检查第一个标签文件对应的图像文件是否存在
    label_file = files[0]
    image_file = os.path.splitext(label_file)[0] + '.jpg'
    image_path = os.path.join(train_dir, image_file)
    print(f'\n检查对应的图像文件: {image_path}')
    if os.path.exists(image_path):
        print('图像文件存在')
    else:
        print('图像文件不存在')

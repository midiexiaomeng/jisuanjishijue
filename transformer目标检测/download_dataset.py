import os
import requests
import tarfile
import zipfile
from tqdm import tqdm
import argparse

def download_file(url, filename):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def extract_tar(file_path, extract_path):
    """解压tar文件"""
    print(f"正在解压 {file_path}...")
    with tarfile.open(file_path, 'r') as tar:
        tar.extractall(extract_path)
    print(f"解压完成到 {extract_path}")

def extract_zip(file_path, extract_path):
    """解压zip文件"""
    print(f"正在解压 {file_path}...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"解压完成到 {extract_path}")

def download_coco_dataset(data_dir="./data/coco"):
    """下载COCO数据集"""
    os.makedirs(data_dir, exist_ok=True)
    
    # COCO数据集URL（使用较小的验证集进行演示）
    urls = {
        "train2017": "http://images.cocodataset.org/zips/train2017.zip",
        "val2017": "http://images.cocodataset.org/zips/val2017.zip",
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    }
    
    print("开始下载COCO数据集...")
    
    for name, url in urls.items():
        filename = os.path.join(data_dir, f"{name}.zip")
        
        if not os.path.exists(filename):
            print(f"下载 {name}...")
            download_file(url, filename)
            
            # 解压文件
            if "annotations" in name:
                extract_zip(filename, data_dir)
            else:
                extract_zip(filename, os.path.join(data_dir, ".."))
        else:
            print(f"{filename} 已存在，跳过下载")
    
    print("COCO数据集下载完成!")

def download_voc_dataset(data_dir="./data/voc"):
    """下载VOC数据集（备选）"""
    os.makedirs(data_dir, exist_ok=True)
    
    # VOC数据集URL
    voc_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    
    print("开始下载VOC数据集...")
    
    filename = os.path.join(data_dir, "VOC2012.tar")
    
    if not os.path.exists(filename):
        print("下载VOC2012...")
        download_file(voc_url, filename)
        
        # 解压文件
        extract_tar(filename, data_dir)
    else:
        print(f"{filename} 已存在，跳过下载")
    
    print("VOC数据集下载完成!")

def download_simple_dataset(data_dir="./data/simple"):
    """下载简单的自定义数据集（用于快速测试）"""
    os.makedirs(data_dir, exist_ok=True)
    
    # 这里可以添加一些简单的测试图像
    # 由于版权问题，我们创建一个简单的模拟数据集结构
    print("创建简单的测试数据集...")
    
    # 创建目录结构
    splits = ['train', 'val']
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # 创建空的标注文件
        annotations_file = os.path.join(data_dir, f"{split}_annotations.json")
        with open(annotations_file, 'w') as f:
            f.write('{"images": [], "annotations": [], "categories": []}')
        
        print(f"创建 {split} 数据集目录: {split_dir}")
    
    print("简单测试数据集创建完成!")
    print("注意：这是一个空的测试数据集结构，您需要添加实际的图像和标注")

def main():
    parser = argparse.ArgumentParser(description="下载目标检测数据集")
    parser.add_argument("--dataset", type=str, default="coco", 
                       choices=["coco", "voc", "simple"],
                       help="要下载的数据集类型")
    parser.add_argument("--data_dir", type=str, default="./data",
                       help="数据保存目录")
    
    args = parser.parse_args()
    
    if args.dataset == "coco":
        download_coco_dataset(os.path.join(args.data_dir, "coco"))
    elif args.dataset == "voc":
        download_voc_dataset(os.path.join(args.data_dir, "voc"))
    elif args.dataset == "simple":
        download_simple_dataset(os.path.join(args.data_dir, "simple"))
    else:
        print(f"未知的数据集类型: {args.dataset}")

if __name__ == "__main__":
    main()

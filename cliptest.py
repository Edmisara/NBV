import os
import torch
from PIL import Image
import clip
import numpy as np

# 标签列表
labels = [
    "table", "chair", "sofa", "bookshelf", "television", "bed", "dining table", 
    "desk", "cupboard", "office chair", "armchair", "leather sofa", "lamp", 
    "cabinet", "refrigerator", "drawer", "wardrobe", "couch", "television stand", 
    "nightstand", "mirror"
]

# 加载CLIP模型和预处理
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

def classify_image(image_path, labels, model, preprocess, device):
    # 打开图片
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)

    # 创建标签输入
    text_inputs = torch.cat([clip.tokenize(label) for label in labels]).to(device)

    # 获取图像和文本的特征
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # 计算图像与每个标签的相似度
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarities = (image_features @ text_features.T).squeeze(0)

    # 获取最匹配标签及其得分
    best_match_idx = similarities.argmax().item()
    best_match_label = labels[best_match_idx]
    best_match_score = similarities[best_match_idx].item()

    return best_match_label, best_match_score

def process_images(image_folder, labels, model, preprocess, device):
    # 获取所有PNG文件
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    results = []
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        best_match_label, best_match_score = classify_image(image_path, labels, model, preprocess, device)
        results.append((image_file, best_match_label, best_match_score))
        print(f"Image: {image_file}, Best Match: {best_match_label}, Score: {best_match_score:.4f}")

    return results

if __name__ == "__main__":
    image_folder = "D:/NBV/nbv_simulation/results/objects"
    results = process_images(image_folder, labels, model, preprocess, device)

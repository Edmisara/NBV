import os
import cv2
import numpy as np
import torch
import open3d as o3d
from PIL import Image
import clip
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# -------------------- 路径设置 --------------------
rgb_path = "D:/NBV/nbv_simulation/results/camera_view.png"
depth_path = "D:/NBV/nbv_simulation/results/depth_image.png"
intrinsic_path = "D:/NBV/nbv_simulation/results/intrinsic_matrix.npy"
extrinsic_path = "D:/NBV/nbv_simulation/results/extrinsic_matrix.npy"
sam_checkpoint = "D:/NBV/nbv_simulation/sam_vit_h_4b8939.pth"
output_dir = "D:/NBV/nbv_simulation/results/objects"
os.makedirs(output_dir, exist_ok=True)

# -------------------- 标签定义 --------------------
labels = [
    "table", "chair", "sofa", "bookshelf", "television", "bed", "dining table",
    "desk", "cupboard", "office chair", "armchair", "leather sofa", "lamp",
    "cabinet", "refrigerator", "drawer", "wardrobe", "couch", "television stand",
    "nightstand", "mirror"
]

# -------------------- 模型加载 --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device)

# -------------------- 工具函数 --------------------
def load_rgb_depth(rgb_path, depth_path):
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    return rgb, depth, rgb.shape[:2]

def run_sam(rgb, sam_checkpoint, model_type="vit_h", device="cuda"):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(rgb)
    return masks

def merge_masks(masks, image_shape, iou_threshold=0.3, min_area_ratio=0.01,
                max_area_ratio=0.8, confidence_threshold=0.5):
    image_area = image_shape[0] * image_shape[1]
    valid_masks = []
    for m in masks:
        seg = m['segmentation'].astype(bool)
        area_ratio = seg.sum() / image_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        if 'predicted_iou' in m and m['predicted_iou'] < confidence_threshold:
            continue
        valid_masks.append(seg)
    return valid_masks

def generate_pointcloud_from_mask(rgb, depth, mask, intrinsic, extrinsic):
    # 注意：depth 是 float32 的 OpenGL 深度值，范围 [0, 1]
    rgb = np.asarray(Image.open(rgb_path))
    depth = np.asarray(Image.open(depth_path)).astype(np.float32) / 255.0  # 还原为 [0, 1] float 深度

    # 加载相机内外参
    intrinsic = np.load(intrinsic_path)  # shape (3, 3)
    extrinsic = np.load(extrinsic_path)  # shape (4, 4)

    # 构造内参对象（需给出图像尺寸和 fx, fy, cx, cy）
    height, width = depth.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # 应用 mask
    mask = mask.astype(bool)
    masked_depth = np.zeros_like(depth)
    masked_depth[mask] = depth[mask]

    masked_rgb = np.zeros_like(rgb)
    masked_rgb[mask] = rgb[mask]

    # 创建 RGBD 图像
    color_o3d = o3d.geometry.Image(masked_rgb.astype(np.uint8))
    depth_o3d = o3d.geometry.Image(masked_depth.astype(np.float32))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1.0,  # 不需要缩放
        depth_trunc=3.0,
        convert_rgb_to_intensity=False
    )

    # 创建点云
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_o3d, extrinsic)

    # 修正坐标系（y轴上下翻转 + z轴反向）
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_o3d, extrinsic)

    # 点云坐标转换
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd

def save_transparent_image(rgb, mask, save_path):
    rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = mask.astype(np.uint8) * 255
    Image.fromarray(rgba).save(save_path)

def classify_image(image_path, labels, model, preprocess, device):
    image = Image.open(image_path)
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(label) for label in labels]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarities = (image_features @ text_features.T).squeeze(0)
    best_match_idx = similarities.argmax().item()
    return labels[best_match_idx], similarities[best_match_idx].item()

# -------------------- 主流程 --------------------
def main():
    rgb, depth, image_shape = load_rgb_depth(rgb_path, depth_path)
    intrinsic = np.load(intrinsic_path)
    extrinsic = np.load(extrinsic_path)

    masks = run_sam(rgb, sam_checkpoint, device=device)
    merged_masks = merge_masks(masks, image_shape)

    print(f"有效 mask 数量: {len(merged_masks)}")
    label_to_masks = {}  # 每类对应的mask列表
    label_to_scores = {}

    for i, mask in enumerate(merged_masks):
        transparent_path = os.path.join(output_dir, f"object_{i:02d}_transparent.png")
        save_transparent_image(rgb, mask, transparent_path)

        label, score = classify_image(transparent_path, labels, clip_model, preprocess, device)
        print(f"[原始] 物体 {i:02d}: {label} ({score:.4f})")

        if label not in label_to_masks:
            label_to_masks[label] = []
            label_to_scores[label] = []
        label_to_masks[label].append(mask)
        label_to_scores[label].append(score)

        # 原始透明图 & 点云输出
        pcd = generate_pointcloud_from_mask(rgb, depth, mask, intrinsic, extrinsic)
        o3d.io.write_point_cloud(os.path.join(output_dir, f"object_{i:02d}_{label}.pcd"), pcd)

    # ------ 合并同类 mask 并分类 ------
    for label, masks_list in label_to_masks.items():
        if len(masks_list) < 2:
            continue  # 不合并只有一个的类

        combined_mask = np.any(np.stack(masks_list), axis=0)
        combined_path = os.path.join(output_dir, f"fused_{label}.png")
        save_transparent_image(rgb, combined_mask, combined_path)

        fused_label, fused_score = classify_image(combined_path, labels, clip_model, preprocess, device)
        original_avg_score = np.mean(label_to_scores[label])
        print(f"[融合] {label}: {fused_label} ({fused_score:.4f}) vs 原始平均得分 {original_avg_score:.4f}")

        if fused_score > original_avg_score and fused_label == label:
            print(f"✅ 使用融合后的 mask 替换 {label}")
            pcd = generate_pointcloud_from_mask(rgb, depth, combined_mask, intrinsic, extrinsic)
            o3d.io.write_point_cloud(os.path.join(output_dir, f"fused_{label}.pcd"), pcd)

if __name__ == "__main__":
    main()

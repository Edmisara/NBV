import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import torch
import json
from PIL import Image
import clip
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# -------------------- 路径设置 --------------------
rgb_path = "D:/NBV/nbv_simulation/results/rgb.png"
depth_path = "D:/NBV/nbv_simulation/results/depth_meter.npy"  # ✅ 修改为 .npy
cam_path = "D:/NBV/nbv_simulation/results/camera.json"
sam_checkpoint = "D:/NBV/nbv_simulation/sam_vit_h_4b8939.pth"
output_dir = "D:/NBV/nbv_simulation/results/objects"
os.makedirs(output_dir, exist_ok=True)

# -------------------- 标签定义 --------------------
labels = [
    "table", "chair", "sofa", "bookshelf", "television", "bed", "dining table",
    "desk", "cupboard", "office chair", "armchair", "leather sofa",
    "cabinet", "refrigerator", "drawer", "wardrobe", "couch", "television stand",
    "nightstand", "mirror"
]

# -------------------- 模型加载 --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device)

# -------------------- 工具函数 --------------------
def load_rgb_depth(rgb_path, depth_path):
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = np.load(depth_path)  # ✅ 直接 npy 加载
    return rgb, depth, rgb.shape[:2]

def run_sam(rgb, sam_checkpoint, model_type="vit_h", device="cuda", max_side=512):
    assert rgb.ndim == 3
    if rgb.shape[2] > 3:
        rgb = rgb[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    h0, w0 = rgb.shape[:2]
    scale = max_side / max(h0, w0)
    resized_rgb = cv2.resize(rgb, (int(h0*scale), int(w0*scale)), interpolation=cv2.INTER_AREA) if scale < 1 else rgb.copy()

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        crop_n_layers=0,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92
    )
    masks = mask_generator.generate(resized_rgb)

    for m in masks:
        m['segmentation'] = cv2.resize(
            m['segmentation'].astype(np.uint8),
            (w0, h0),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    return masks

def merge_masks(masks, image_shape, iou_threshold=0.3, containment_threshold=0.9,
                min_area_ratio=0.01, max_area_ratio=0.8, confidence_threshold=0.5):
    image_area = image_shape[0] * image_shape[1]
    filtered_masks = []

    for m in masks:
        seg = m['segmentation'].astype(bool)
        area_ratio = seg.sum() / image_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        if 'predicted_iou' in m and m['predicted_iou'] < confidence_threshold:
            continue
        filtered_masks.append(seg)

    merged = []
    used = set()

    for i in range(len(filtered_masks)):
        if i in used:
            continue
        current = filtered_masks[i]
        to_merge = [current]

        for j in range(i + 1, len(filtered_masks)):
            if j in used:
                continue
            candidate = filtered_masks[j]
            intersection = np.logical_and(current, candidate).sum()
            union = np.logical_or(current, candidate).sum()
            if union == 0:
                continue
            iou = intersection / union
            containment = max(intersection / current.sum(), intersection / candidate.sum())

            if iou > iou_threshold or containment > containment_threshold:
                to_merge.append(candidate)
                used.add(j)

        merged_mask = np.any(to_merge, axis=0)
        merged.append(merged_mask)

    return merged


def save_pointcloud_as_ply(points, colors, filename):
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")


def generate_pointcloud_from_mask(rgb, depth, mask, cam_path):
    """
    使用逐像素for循环反投影（与cloud.ply生成方法完全一致），
    并应用mask筛选有效点。
    """
    height, width = depth.shape

    # === 加载相机参数 ===
    with open(cam_path, "r") as f:
        cam = json.load(f)

    intrinsic = cam["intrinsic"]
    fx, fy, cx, cy = intrinsic["fx"], intrinsic["fy"], intrinsic["cx"], intrinsic["cy"]
    T_world2cam = np.array(cam["extrinsic"])
    T_cam2world = np.linalg.inv(T_world2cam)

    points = []
    colors = []

    for v in range(height):
        for u in range(width):
            z = depth[v, u]
            if z == 0:
                continue
            if mask[v, u] == 0:
                continue

            # ✅ 使用单位射线
            x = (u - cx) / fx
            y = -(v - cy) / fy
            ray = np.array([x, y, -1.0])
            ray /= np.linalg.norm(ray)

            point_cam = ray * z

            # ✅ 转为世界坐标
            cam_point_h = np.concatenate([point_cam, [1.0]])
            world_point = T_cam2world @ cam_point_h
            points.append(world_point[:3])

            # ✅ 同步颜色
            if rgb is not None:
                r, g, b = rgb[v, u]
                colors.append((r, g, b))

    points = np.array(points, dtype=np.float32)
    colors = np.array(colors, dtype=np.uint8)

    return points, colors

def save_transparent_image(rgb, mask, save_path):
    if rgb.shape[2] > 3:
        rgb = rgb[:, :, :3]
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

    masks = run_sam(rgb, sam_checkpoint, device=device, max_side=512)
    merged_masks = merge_masks(masks, image_shape)

    print(f"有效 mask 数量: {len(merged_masks)}")
    label_to_masks = {}
    label_to_scores = {}

    for i, mask in enumerate(merged_masks):
        transparent_path = os.path.join(output_dir, f"temp_object_{i:02d}_transparent.png")
        save_transparent_image(rgb, mask, transparent_path)

        label, score = classify_image(transparent_path, labels, clip_model, preprocess, device)
        print(f"[原始] 物体 {i:02d}: {label} ({score:.4f})")

        if label not in label_to_masks:
            label_to_masks[label] = []
            label_to_scores[label] = []
        label_to_masks[label].append(mask)
        label_to_scores[label].append(score)

        os.remove(transparent_path)

    for label, masks_list in label_to_masks.items():
        if len(masks_list) >= 2:
            combined_mask = np.any(np.stack(masks_list), axis=0)
            combined_path = os.path.join(output_dir, f"fused_{label}.png")
            save_transparent_image(rgb, combined_mask, combined_path)

            fused_label, fused_score = classify_image(combined_path, labels, clip_model, preprocess, device)
            original_avg_score = np.mean(label_to_scores[label])
            print(f"[融合] {label}: {fused_label} ({fused_score:.4f}) vs 原始平均得分 {original_avg_score:.4f}")

            if fused_score > original_avg_score and fused_label == label:
                print(f"✅ 使用融合后的 mask 输出 {label}")
                points, colors = generate_pointcloud_from_mask(rgb, depth, combined_mask, cam_path)
                save_pointcloud_as_ply(points, colors, os.path.join(output_dir, f"fused_{label}.ply"))
            else:
                print(f"⚠️ 融合效果不佳，跳过 {label}")
        else:
            mask = masks_list[0]
            combined_path = os.path.join(output_dir, f"fused_{label}.png")
            save_transparent_image(rgb, mask, combined_path)

            print(f"[无融合] 输出 {label}")
            points, colors = generate_pointcloud_from_mask(rgb, depth, mask, cam_path)
            save_pointcloud_as_ply(points, colors, os.path.join(output_dir, f"fused_{label}.ply"))

if __name__ == "__main__":
    main()

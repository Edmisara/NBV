import numpy as np
import cv2
import os
import torch
import open3d as o3d
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image

def load_rgb_depth(rgb_path, depth_path):
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    return rgb, depth

def generate_pointcloud_from_mask(rgb, depth, mask, intrinsic, extrinsic, depth_scale=1000.0):
    h, w = depth.shape
    # Create Open3D RGBD image
    mask = mask.astype(bool)
    
    # 只保留 mask 区域的深度和 RGB
    masked_depth = np.zeros_like(depth)
    masked_depth[mask] = depth[mask]

    # 用透明背景填充 RGB
    masked_rgb = np.zeros_like(rgb)
    masked_rgb[mask] = rgb[mask]

    color_o3d = o3d.geometry.Image(masked_rgb)
    depth_o3d = o3d.geometry.Image(masked_depth.astype(np.uint16))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=depth_scale,
        convert_rgb_to_intensity=False
    )

    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
    intrinsic_o3d.intrinsic_matrix = intrinsic

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_o3d, extrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # Flip for Open3D
    return pcd

def save_transparent_image(rgb, mask, save_path):
    # Create a transparent background image (RGBA format)
    rgba_image = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    rgba_image[..., :3] = rgb  # Copy RGB channels
    rgba_image[..., 3] = mask * 255  # Alpha channel (opaque where mask is True)

    # Save the image with transparent background
    transparent_img = Image.fromarray(rgba_image)
    transparent_img.save(save_path)
    print(f"透明背景图已保存为 {save_path}")

def run_sam(rgb, sam_checkpoint, model_type="vit_h", device="cuda"):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(rgb)
    return masks

def main():
    # 路径设置
    rgb_path = "D:\\NBV\\nbv_simulation\\results\\camera_view.png"
    depth_path = "D:\\NBV\\nbv_simulation\\results\\depth_image.png"
    intrinsic_path = "D:\\NBV\\nbv_simulation\\results\\intrinsic_matrix.npy"
    extrinsic_path = "D:\\NBV\\nbv_simulation\\results\\extrinsic_matrix.npy"
    sam_checkpoint = "D:\\NBV\\nbv_simulation\\sam_vit_h_4b8939.pth"  # 请下载 SAM 的 checkpoint 文件
    output_dir = "D:\\NBV\\nbv_simulation\\results\\objects"

    os.makedirs(output_dir, exist_ok=True)

    # 加载图像和参数
    rgb, depth = load_rgb_depth(rgb_path, depth_path)
    intrinsic = np.load(intrinsic_path)
    extrinsic = np.load(extrinsic_path)

    # 运行 SAM 分割
    masks = run_sam(rgb, sam_checkpoint)

    print(f"SAM 生成了 {len(masks)} 个 mask")

    # 遍历每个 mask，生成点云和透明背景图
    for i, m in enumerate(masks):
        mask = m['segmentation']
        pcd = generate_pointcloud_from_mask(rgb, depth, mask, intrinsic, extrinsic)
        
        # 保存点云
        pcd_filename = os.path.join(output_dir, f"object_{i:02d}.pcd")
        o3d.io.write_point_cloud(pcd_filename, pcd)
        print(f"已保存点云：{pcd_filename}")

        # 保存透明背景的 RGB 图
        transparent_filename = os.path.join(output_dir, f"object_{i:02d}_transparent.png")
        save_transparent_image(rgb, mask, transparent_filename)

if __name__ == "__main__":
    main()
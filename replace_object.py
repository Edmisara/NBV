import os
import numpy as np
import open3d as o3d
import json

def load_camera_position(camera_json_path):
    with open(camera_json_path, 'r') as f:
        cam_data = json.load(f)
    extrinsic = np.array(cam_data['extrinsic'])  # T_world2cam
    cam_position = np.linalg.inv(extrinsic)[:3, 3]  # 从世界到相机，取逆得到相机位置
    return cam_position

def compute_pca(points):
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return eigvecs[:, order]

def compute_bbox_size(points):
    min_pt = points.min(axis=0)
    max_pt = points.max(axis=0)
    return max_pt - min_pt

def replace_object(fused_pcd_path, matched_model_pcd_path, cloud_pcd_path, camera_json_path, output_pcd_path):
    # 加载点云
    fused_pcd = o3d.io.read_point_cloud(fused_pcd_path)
    model_pcd = o3d.io.read_point_cloud(matched_model_pcd_path)
    cloud_pcd = o3d.io.read_point_cloud(cloud_pcd_path)

    fused_points = np.asarray(fused_pcd.points)
    model_points = np.asarray(model_pcd.points)
    cloud_points = np.asarray(cloud_pcd.points)

    # 1. 相机位置
    cam_pos = load_camera_position(camera_json_path)

    # 2. 找最近点方向
    dists = np.linalg.norm(fused_points - cam_pos, axis=1)
    nearest_idx = np.argsort(dists)[:max(10, int(0.05 * len(fused_points)))]
    nearest_points = fused_points[nearest_idx]
    nearest_center = nearest_points.mean(axis=0)
    dir_vec = nearest_center - cam_pos
    dir_vec /= np.linalg.norm(dir_vec)

    # 3. 推远一定距离（设为局部bbox对角线的0.5倍）
    fused_bbox_size = compute_bbox_size(fused_points)
    delta = 0.5 * np.linalg.norm(fused_bbox_size)
    target_center = nearest_center + delta * dir_vec

    # 4. 尺寸缩放
    model_bbox_size = compute_bbox_size(model_points)
    scale_factors = fused_bbox_size / (model_bbox_size + 1e-6)
    scale = np.mean(scale_factors)
    model_points_scaled = model_points * scale

    # 5. 朝向对齐（PCA）
    fused_pca = compute_pca(fused_points)
    model_pca = compute_pca(model_points_scaled)

    R = fused_pca @ model_pca.T
    model_points_aligned = (R @ model_points_scaled.T).T

    # 6. 平移到目标中心
    model_points_transformed = model_points_aligned - model_points_aligned.mean(axis=0) + target_center

    model_pcd_transformed = o3d.geometry.PointCloud()
    model_pcd_transformed.points = o3d.utility.Vector3dVector(model_points_transformed)
    if model_pcd.has_colors():
        model_pcd_transformed.colors = model_pcd.colors

    # 7. 从cloud中删除原有的fused_table区域
    fused_kd_tree = o3d.geometry.KDTreeFlann(fused_pcd)
    mask = np.ones(len(cloud_points), dtype=bool)
    for i, pt in enumerate(cloud_points):
        [_, idx, _] = fused_kd_tree.search_knn_vector_3d(pt, 1)
        nearest_pt = np.asarray(fused_pcd.points)[idx[0]]
        if np.linalg.norm(pt - nearest_pt) < 0.02:  # 距离阈值可调
            mask[i] = False
    cloud_points_filtered = cloud_points[mask]
    cloud_pcd_filtered = o3d.geometry.PointCloud()
    cloud_pcd_filtered.points = o3d.utility.Vector3dVector(cloud_points_filtered)

    # 8. 合并并保存
    final_pcd = cloud_pcd_filtered + model_pcd_transformed
    o3d.io.write_point_cloud(output_pcd_path, final_pcd)
    print(f"✅ 替换完成，保存到 {output_pcd_path}")

if __name__ == "__main__":
    fused_pcd_path = "D:/NBV/nbv_simulation/results/objects/fused_table.ply"
    matched_model_pcd_path = "D:/NBV/nbv_simulation/data/ply/table_001.ply"
    cloud_pcd_path = "D:/NBV/nbv_simulation/results/cloud.ply"
    camera_json_path = "D:/NBV/nbv_simulation/results/camera.json"
    output_pcd_path = "D:/NBV/nbv_simulation/results/final_replaced_cloud.ply"

    replace_object(fused_pcd_path, matched_model_pcd_path, cloud_pcd_path, camera_json_path, output_pcd_path)

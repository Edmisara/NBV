import open3d as o3d
import numpy as np

# ==== 配置路径 ====
query_path = "D:/NBV/nbv_simulation/results/objects/fused_table.ply"
model_path = "D:/NBV/nbv_simulation/data/ply/table_005.ply"

# ==== 可调参数 ====
voxel_size = 0.05
radius_normal = voxel_size * 2
radius_feature = voxel_size * 5

# ==== 工具函数区 ====
def print_pcd_info(name, pcd):
    pts = np.asarray(pcd.points)
    print(f"\n[INFO] {name}:")
    print(f"  点数: {pts.shape[0]}")
    print(f"  min坐标: {pts.min(axis=0)}")
    print(f"  max坐标: {pts.max(axis=0)}")
    print(f"  跨度 (max - min): {pts.max(axis=0) - pts.min(axis=0)}")

def normalize_pcd(pcd):
    pts = np.asarray(pcd.points)
    center = pts.mean(axis=0)
    scale = np.linalg.norm(pts - center, axis=1).max()
    pts = (pts - center) / scale
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd, center, scale

def preprocess_with_trace(pcd):
    down, _, indices = pcd.voxel_down_sample_and_trace(
        voxel_size=voxel_size,
        min_bound=pcd.get_min_bound() - voxel_size * 0.5,
        max_bound=pcd.get_max_bound() + voxel_size * 0.5,
        approximate_class=False)

    down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius_normal, max_nn=30))
    
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return down, fpfh, indices

def estimate_affine(source_points, target_points):
    """
    基于最小二乘估计仿射矩阵，包括scale+rotation+translation
    """

    # === 添加一列1，扩展成齐次坐标 ===
    num_points = source_points.shape[0]
    source_h = np.hstack([source_points, np.ones((num_points, 1))])  # (N,4)

    # === 直接解线性系统 ===
    # 目标是求一个 (3x4) 矩阵，使得 source_h @ A.T ≈ target_points
    A, residuals, rank, s = np.linalg.lstsq(source_h, target_points, rcond=None)

    # === 组装成4x4仿射矩阵 ===
    T = np.eye(4)
    T[:3, :] = A.T

    return T
# ==== 主流程 ====

def main():
    # === 加载原始点云 ===
    query_full = o3d.io.read_point_cloud(query_path)
    model_full = o3d.io.read_point_cloud(model_path)
    print_pcd_info("原始 query_full (fused_table)", query_full)

    # === ⚡ 立即clone一份query_full_ori保护 ===
    query_full_ori = o3d.io.read_point_cloud(query_path)
    model_full_ori = o3d.io.read_point_cloud(model_path)

    # === 归一化（操作query_full，不污染query_full_ori） ===
    query_norm, query_center, query_scale = normalize_pcd(query_full)
    model_norm, _, _ = normalize_pcd(model_full)

    # === voxel下采样并记录trace ===
    query_down, query_fpfh, query_indices = preprocess_with_trace(query_norm)
    model_down, model_fpfh, model_indices = preprocess_with_trace(model_norm)

    # === 配准（RANSAC + ICP） ===
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        query_down, model_down, query_fpfh, model_fpfh, True, voxel_size * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 500))

    result_icp = o3d.pipelines.registration.registration_icp(
        query_down, model_down, voxel_size * 1.5, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    print(f"[LOG] 配准完成，fitness: {result_icp.fitness:.4f}")

    # === 可视化 ①：归一化配准的小点 ===
    query_down.paint_uniform_color([1, 0, 0])  # 红
    model_down.paint_uniform_color([0, 1, 0])  # 绿
    query_down.transform(result_icp.transformation)

    o3d.visualization.draw_geometries([query_down, model_down])

    # === 用稀疏点索引还原原始点 ===
    src_pts = []
    tgt_pts = []
    for idx_query, idx_model in np.asarray(result_icp.correspondence_set):
        if query_indices[idx_query] and model_indices[idx_model]:
            src_idx = model_indices[idx_model][0]
            tgt_idx = query_indices[idx_query][0]
            src_pts.append(np.asarray(model_full_ori.points)[src_idx])
            tgt_pts.append(np.asarray(query_full_ori.points)[tgt_idx])  # ⚡ 注意用query_full_ori

    src_pts = np.array(src_pts)
    tgt_pts = np.array(tgt_pts)

    # === 基于原始点计算仿射矩阵 ===
    T_affine = estimate_affine(src_pts, tgt_pts)

    # === 应用仿射到完整obj ===
    model_full_transformed = model_full_ori.transform(T_affine)

    # === 打印检查 ===
    print_pcd_info("仿射后的 model_full_transformed", model_full_transformed)
    print_pcd_info("query_full_ori", query_full_ori)

    # === 可视化 ②：原始空间完整对齐结果 ===
    query_full_ori.paint_uniform_color([1, 0, 0])  # 红
    model_full_transformed.paint_uniform_color([0, 1, 0])  # 绿

    o3d.visualization.draw_geometries([query_full_ori, model_full_transformed])
    print("\n[LOG] 开始替换cloud...")

    # 重新加载 cloud
    cloud = o3d.io.read_point_cloud("D:/NBV/nbv_simulation/results/cloud.ply")
    # fused_table 直接用 query_full_ori

    # 构建 KDTree
    fused_tree = o3d.geometry.KDTreeFlann(query_full_ori)

    # 设定删除半径（比如 1cm）
    threshold = 0.5

    cloud_points = np.asarray(cloud.points)
    mask = []

    for pt in cloud_points:
        [k, _, _] = fused_tree.search_radius_vector_3d(pt, threshold)
        mask.append(k == 0)  # 保留远离fused_table的点

    mask = np.array(mask)

    # 保存【遮挡物】cloud_clean
    cloud_clean = o3d.geometry.PointCloud()
    cloud_clean.points = o3d.utility.Vector3dVector(cloud_points[mask])
    gray_clean = np.full_like(np.asarray(cloud_clean.points), 0.5)  # 灰色 (0.5, 0.5, 0.5)
    cloud_clean.colors = o3d.utility.Vector3dVector(gray_clean)

    o3d.io.write_point_cloud("D:/NBV/nbv_simulation/results/cloud_clean.ply", cloud_clean)
    print("[LOG] cloud_clean.ply 已保存。")

    # 保存【近似物体】model_full_transformed
    gray_model = np.full_like(np.asarray(model_full_transformed.points), 0.5)
    model_full_transformed.colors = o3d.utility.Vector3dVector(gray_model)

    o3d.io.write_point_cloud("D:/NBV/nbv_simulation/results/model_full_transformed.ply", model_full_transformed)
    print("[LOG] model_full_transformed.ply 已保存。")

    # 仅用于可视化（合并）
    combined = cloud_clean + model_full_transformed
    o3d.visualization.draw_geometries([combined])


if __name__ == "__main__":
    main()

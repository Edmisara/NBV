import open3d as o3d
import numpy as np
import os

def estimate_eps(pcd, nb_neighbors=5, scale_factor=2.0):
    """
    自动根据最近邻均距估计eps参数。
    """
    print("[LOG] 计算最近邻距离...")
    distances = pcd.compute_nearest_neighbor_distance()
    distances = np.array(distances)

    # 排除极端大值，防止飞点干扰
    sorted_distances = np.sort(distances)
    cutoff = int(len(sorted_distances) * 0.9)  # 取前90%
    avg_distance = np.mean(sorted_distances[:cutoff])

    eps = avg_distance * scale_factor
    print(f"[LOG] 最近邻均距: {avg_distance:.6f}m, 自动推eps: {eps:.6f}m")
    return eps

def clean_point_cloud_by_clustering(pcd, eps, min_points=10):
    """
    使用DBSCAN提取最大连通块，清理离群点。
    """
    print("[LOG] 开始DBSCAN聚类...")
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))

    if labels.max() < 0:
        print("[Warning] 没有找到任何连通块，返回原始点云。")
        return pcd

    largest_cluster_idx = np.argmax(np.bincount(labels[labels >= 0]))
    indices = np.where(labels == largest_cluster_idx)[0]

    print(f"[LOG] 最大连通块点数: {len(indices)} / 总点数: {len(pcd.points)}")
    pcd_clean = pcd.select_by_index(indices)
    return pcd_clean

def main():
    input_path = "D:/NBV/nbv_simulation/results/objects/fused_table.ply"  # 修改为你的输入路径
    output_path_clean = "D:/NBV/nbv_simulation/results/objects/fused_table_clean.ply"

    # === 加载点云 ===
    print(f"[LOG] 加载点云: {input_path}")
    pcd = o3d.io.read_point_cloud(input_path)

    # === 自动估计eps ===
    eps = estimate_eps(pcd, nb_neighbors=10, scale_factor=4.0)

    # === 聚类清理 ===
    pcd_clean = clean_point_cloud_by_clustering(pcd, eps=eps, min_points=10)

    # === 保存清理后的点云 ===
    o3d.io.write_point_cloud(input_path, pcd_clean)
    print(f"[LOG] 保存清理后点云: {input_path}")

    # === 可视化对比 ===
    pcd.paint_uniform_color([1, 0, 0])       # 原始-红色
    pcd_clean.paint_uniform_color([0, 1, 0]) # 清理后-绿色
    o3d.visualization.draw_geometries([pcd, pcd_clean])

if __name__ == "__main__":
    main()

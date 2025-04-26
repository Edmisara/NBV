import open3d as o3d

# 加载点云文件
pcd_path = "D:/NBV/nbv_simulation/results/objects/fused_table.pcd"
pcd = o3d.io.read_point_cloud(pcd_path)

# 打印信息
print(f"✅ 点数: {len(pcd.points)}")
bbox = pcd.get_axis_aligned_bounding_box()
print(f"📦 坐标范围 (单位: 原始):")
print(f"X: {bbox.min_bound[0]:.4f} → {bbox.max_bound[0]:.4f}")
print(f"Y: {bbox.min_bound[1]:.4f} → {bbox.max_bound[1]:.4f}")
print(f"Z: {bbox.min_bound[2]:.4f} → {bbox.max_bound[2]:.4f}")

# 显示点云
o3d.visualization.draw_geometries([pcd])
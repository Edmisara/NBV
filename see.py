import open3d as o3d
import numpy as np

# === 文件路径 ===
mesh_path = "D:/NBV/nbv_simulation/data/table_001.obj"
fused_pcd_path = "D:/NBV/nbv_simulation/results/objects/fused_table.pcd"

# === 加载 mesh 和点云 ===
mesh = o3d.io.read_triangle_mesh(mesh_path)
mesh.compute_vertex_normals()
fused_pcd = o3d.io.read_point_cloud(fused_pcd_path)

# === 应用最佳匹配变换 ===
scale = 0.95
dx, dy, dz = -150.0, -150.0, -150.0

mesh.scale(scale, center=mesh.get_center())
mesh.translate((dx, dy, dz), relative=False)

# === 上色：mesh 蓝色，fused_pcd 绿色 ===
mesh.paint_uniform_color([0.0, 0.6, 1.0])
fused_pcd.paint_uniform_color([0.0, 1.0, 0.0])

# === 可视化结果 ===
o3d.visualization.draw_geometries(
    [mesh, fused_pcd],
    window_name="OBJ 与 Fused 点云对齐结果",
    width=960,
    height=720
)

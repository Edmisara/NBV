import os
import pickle
import open3d as o3d
import numpy as np
from collections import defaultdict

# ==== 路径设置 ==== 
obj_folder = "D:/NBV/nbv_simulation/data"
save_ply_folder = os.path.join(obj_folder, "ply")  # ✅ 改成 ply 文件夹
os.makedirs(save_ply_folder, exist_ok=True)

library_output_path = os.path.join(obj_folder, "furniture_library.pkl")
labels_txt_path = os.path.join(obj_folder, "labels.txt")

# ==== 边缘增强函数 ==== 
def enhance_edges(pcd, k=30, curvature_threshold=0.08):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    curvatures = []
    pts = np.asarray(pcd.points)
    tree = o3d.geometry.KDTreeFlann(pcd)
    for i in range(len(pts)):
        [_, idx, _] = tree.search_knn_vector_3d(pcd.points[i], k)
        neighbors = pts[idx[1:], :]
        cov = np.cov((neighbors - pts[i]).T)
        eigvals, _ = np.linalg.eigh(cov)
        curvature = eigvals[0] / (eigvals.sum() + 1e-8)
        curvatures.append(curvature)
    curvatures = np.array(curvatures)
    edge_idx = np.where(curvatures > curvature_threshold)[0]
    edge_points = pts[edge_idx]

    duplicated = o3d.geometry.PointCloud()
    duplicated.points = o3d.utility.Vector3dVector(np.repeat(edge_points, 3, axis=0))
    enhanced_pcd = pcd + duplicated
    return enhanced_pcd.voxel_down_sample(voxel_size=0.002)

# ==== 采样函数（带兜底 + 增强） ==== 
def safe_sample_mesh(mesh, num_points=8192):
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    try:
        mesh = mesh.subdivide_midpoint(number_of_iterations=1)
        pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
    except:
        print("⚠️ Poisson采样失败，尝试Uniform采样")
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)

    if len(pcd.points) == 0:
        return pcd

    pcd = enhance_edges(pcd)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(50)
    return pcd

# ==== 主逻辑 ==== 
model_dict = defaultdict(list)

for fname in os.listdir(obj_folder):
    if not fname.endswith(".obj"):
        continue
    label = fname.split("_")[0]
    obj_path = os.path.join(obj_folder, fname)
    mesh = o3d.io.read_triangle_mesh(obj_path)
    if not mesh.has_triangles():
        print(f"[跳过] {fname} 无有效三角面")
        continue

    pcd = safe_sample_mesh(mesh)
    if len(pcd.points) == 0:
        print(f"[跳过] {fname} 采样失败，无点云")
        continue

    # 保存 .ply
    ply_filename = fname.replace(".obj", ".ply")
    ply_path = os.path.join(save_ply_folder, ply_filename)
    o3d.io.write_point_cloud(ply_path, pcd)  # ✅ 保存为 ply

    # 加入模型字典（保存路径而非点云对象）
    model_dict[label].append((ply_filename, ply_path))
    print(f"✅ 已处理并保存: {fname} → {ply_filename}")

# ==== 保存为新的 .pkl 库 ==== 
with open(library_output_path, "wb") as f:
    pickle.dump(model_dict, f)
print("\n✅ 点云模型路径库已重新构建并保存为:", library_output_path)

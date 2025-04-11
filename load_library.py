import os
import pickle
import open3d as o3d
from collections import defaultdict

# ==== 路径设置 ====
obj_folder = "D:/NBV/nbv_simulation/data"
save_pcd_folder = os.path.join(obj_folder, "pcd")
os.makedirs(save_pcd_folder, exist_ok=True)

library_output_path = os.path.join(obj_folder, "furniture_library.pkl")
labels_txt_path = os.path.join(obj_folder, "labels.txt")

# ==== 采样函数（带兜底） ====
def safe_sample_mesh(mesh, num_points=2048):
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    try:
        pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
    except:
        print("⚠️ Poisson 采样失败，尝试 Uniform 采样")
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)

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

    # 保存 .pcd
    pcd_filename = fname.replace(".obj", ".pcd")
    pcd_path = os.path.join(save_pcd_folder, pcd_filename)
    o3d.io.write_point_cloud(pcd_path, pcd)

    # 加入模型字典（保存路径而非点云对象）
    model_dict[label].append((fname, pcd_path))
    print(f"✅ 已处理并保存: {fname} → {pcd_filename}")

# ==== 保存为新的 .pkl 库 ====
with open(library_output_path, "wb") as f:
    pickle.dump(model_dict, f)
print("\n✅ 点云模型路径库已重新构建并保存为:", library_output_path)
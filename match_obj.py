import os
import pickle
import numpy as np
import open3d as o3d

# ===== 用户路径设置 =====
fused_pcd_path = "D:/NBV/nbv_simulation/results/objects/fused_table.pcd"
furniture_library_path = "D:/NBV/nbv_simulation/data/furniture_library.pkl"
model_obj_folder = "D:/NBV/nbv_simulation/data"

# ===== 加载目标点云并解析标签 =====
query_pcd = o3d.io.read_point_cloud(fused_pcd_path)
label = os.path.basename(fused_pcd_path).replace("fused_", "").replace(".pcd", "")
print(f"[1] 加载查询点云: {fused_pcd_path}\n     → 标签: {label}")

# ===== 点云归一化（中心化 + 缩放） =====
def normalize_pcd(pcd):
    pts = np.asarray(pcd.points)
    pts -= pts.mean(axis=0)
    scale = np.linalg.norm(pts, axis=1).max()
    pts /= scale
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

# ===== 特征计算函数 =====
def preprocess_point_cloud(pcd, voxel_size):
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return pcd, fpfh

# ===== 粗配准 + ICP 精配准 =====
def global_icp_match(source, target, voxel_size=0.05):
    source = normalize_pcd(source)
    target = normalize_pcd(target)
    src_down, src_fpfh = preprocess_point_cloud(source, voxel_size)
    tgt_down, tgt_fpfh = preprocess_point_cloud(target, voxel_size)

    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh, True, voxel_size * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))

    result_icp = o3d.pipelines.registration.registration_icp(
        source, target, voxel_size * 1.5, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    return result_icp.fitness, result_icp.inlier_rmse

# ===== 加载模型点云路径库 =====
with open(furniture_library_path, "rb") as f:
    model_dict = pickle.load(f)

if label not in model_dict:
    print(f"❌ 标签 {label} 不在模型库中")
    exit(1)

# ===== 匹配同类模型 =====
best_model = None
best_fitness = 0
best_model_path = None

for fname, pcd_path in model_dict[label]:
    try:
        model_pcd = o3d.io.read_point_cloud(pcd_path)
        fitness, rmse = global_icp_match(query_pcd, model_pcd)
        print(f"[匹配] {fname} → fitness={fitness:.4f}, rmse={rmse:.4f}")
        if fitness > best_fitness:
            best_fitness = fitness
            best_model = fname
            best_model_path = os.path.join(model_obj_folder, fname)
    except Exception as e:
        print(f"[跳过] {fname} 匹配失败：{e}")

if best_model:
    print("\n✅ 匹配成功")
    print(f"最相似模型: {best_model}")
    print(f"fitness: {best_fitness:.4f}")
    print(f"OBJ路径: {best_model_path}")
else:
    print("❌ 未找到匹配模型")
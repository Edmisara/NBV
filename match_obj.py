import os
import pickle
import numpy as np
import open3d as o3d
import multiprocessing as mp
import cv2

# ==== 可调参数区 ====
voxel_size = 0.05
radius_normal = voxel_size * 2
radius_feature = voxel_size * 5
lambda_weight = 3.0
gamma_weight = 0.2
xi_weight = 0.8
baseline_corr = 3000
apply_downsample = True

query_folder = "D:/NBV/nbv_simulation/results/objects"
furniture_library_path = "D:/NBV/nbv_simulation/data/furniture_library.pkl"
model_ply_folder = "D:/NBV/nbv_simulation/data/ply"

# ==== normalize + preprocess ====
def normalize_pcd(pcd):
    pts = np.asarray(pcd.points)
    pts -= pts.mean(axis=0)
    scale = np.linalg.norm(pts, axis=1).max()
    pts /= scale
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def preprocess_point_cloud(pcd):
    if apply_downsample:
        pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd, fpfh

def extract_shape_descriptor(pcd, n=30):
    points = np.asarray(pcd.points)
    points -= points.mean(axis=0)
    scale = np.linalg.norm(points, axis=1).max()
    points /= scale
    projected = points[:, [0, 2]]
    projected += 1
    projected *= 256
    projected = projected.astype(np.int32)
    mask = np.zeros((512, 512), dtype=np.uint8)
    for pt in projected:
        if 0 <= pt[0] < 512 and 0 <= pt[1] < 512:
            mask[pt[1], pt[0]] = 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.ones(n)
    cnt = max(contours, key=cv2.contourArea)
    cnt = cnt.squeeze()
    if len(cnt.shape) != 2 or cnt.shape[0] < 16:
        return np.ones(n)
    complex_cnt = cnt[:, 0] + 1j * cnt[:, 1]
    complex_cnt -= complex_cnt.mean()
    fourier = np.fft.fft(complex_cnt)
    descriptor = np.abs(fourier[1:n+1])
    descriptor /= descriptor[0] + 1e-6
    return descriptor

def match_worker(args):
    query_down_np, query_fpfh_np, query_desc, fname, model_down_np, model_fpfh_np, model_desc = args
    try:
        query_down = o3d.geometry.PointCloud()
        query_down.points = o3d.utility.Vector3dVector(query_down_np)
        query_fpfh = o3d.pipelines.registration.Feature()
        query_fpfh.data = query_fpfh_np

        model_down = o3d.geometry.PointCloud()
        model_down.points = o3d.utility.Vector3dVector(model_down_np)
        model_fpfh = o3d.pipelines.registration.Feature()
        model_fpfh.data = model_fpfh_np

        result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            query_down, model_down, query_fpfh, model_fpfh, True, voxel_size * 1.5,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 500))

        result_icp = o3d.pipelines.registration.registration_icp(
            query_down, model_down, voxel_size * 1.5, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())

        fitness = result_icp.fitness
        rmse = result_icp.inlier_rmse
        corr = len(result_icp.correspondence_set)

        shape_score = 1 - np.linalg.norm(model_desc - query_desc) / np.sqrt(len(query_desc))
        shape_score = float(np.clip(shape_score, 0.0, 1.0))

                # ====== 可视化匹配结果（仅在fitness高于0.9时显示）======
        if fitness is not None and fitness > 0.9:
            print(f"[可视化] {fname} 匹配fitness高，显示一下结果")
            query_vis = o3d.geometry.PointCloud()
            query_vis.points = o3d.utility.Vector3dVector(query_down_np)
            query_vis.paint_uniform_color([1, 0, 0])  # 红色 - 查询

            model_vis = o3d.geometry.PointCloud()
            model_vis.points = o3d.utility.Vector3dVector(model_down_np)
            model_vis.paint_uniform_color([0, 1, 0])  # 绿色 - 匹配

            query_vis.transform(result_icp.transformation)

            o3d.visualization.draw_geometries([query_vis, model_vis])
            
        return fname, fitness, rmse, corr, shape_score
    except Exception as e:
        return fname, None, None, None, None

# ==== 主流程 ====
def match_one_query(fused_ply_path, model_dict):
    # === 加载查询点云 ===
    query_pcd = o3d.io.read_point_cloud(fused_ply_path)
    label = os.path.basename(fused_ply_path).replace("fused_", "").replace(".ply", "")
    print(f"\n[1] 加载查询点云: {fused_ply_path}\n     → 标签: {label}")

    if label not in model_dict:
        print(f"❌ 标签 {label} 不在模型库中，跳过。")
        return

    query_pcd = normalize_pcd(query_pcd)
    query_down, query_fpfh = preprocess_point_cloud(query_pcd)
    query_down_np = np.asarray(query_down.points)
    query_fpfh_np = query_fpfh.data
    query_desc = extract_shape_descriptor(query_down)

    tasks = []
    for fname, ply_path in model_dict[label]:
        try:
            full_path = os.path.join(model_ply_folder, fname)
            model_pcd = o3d.io.read_point_cloud(full_path)
            model_pcd = normalize_pcd(model_pcd)
            model_down, model_fpfh = preprocess_point_cloud(model_pcd)
            model_down_np = np.asarray(model_down.points)
            model_fpfh_np = model_fpfh.data
            model_desc = extract_shape_descriptor(model_down)
            tasks.append((query_down_np, query_fpfh_np, query_desc, fname, model_down_np, model_fpfh_np, model_desc))
        except Exception as e:
            print(f"[跳过] {fname} 加载失败：{e}")

    best_model = None
    best_score = -np.inf
    best_model_path = None
    best_fitness = 0

    print("🚀 启动多进程匹配...")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(match_worker, tasks)

    for fname, fitness, rmse, corr, shape_score in results:
        if fitness is None:
            print(f"[跳过] {fname} 匹配失败。")
            continue
        score = fitness - lambda_weight * rmse + gamma_weight * (corr / baseline_corr) + xi_weight * shape_score
        print(f"[匹配] {fname} → fitness={fitness:.4f}, rmse={rmse:.4f}, corr={corr}, shape_score={shape_score:.4f}, score={score:.4f}")
        if score > best_score:
            best_score = score
            best_fitness = fitness
            best_model = fname
            best_model_path = os.path.join(model_ply_folder, fname)

    if best_model:
        print(f"\n✅ 匹配成功：{best_model}")
        print(f"fitness: {best_fitness:.4f}")
        print(f"score: {best_score:.4f}")
        print(f"PLY路径: {best_model_path}")
    else:
        print("❌ 未找到匹配模型")

def main():
    with open(furniture_library_path, "rb") as f:
        model_dict = pickle.load(f)

    ply_files = [os.path.join(query_folder, fname) for fname in os.listdir(query_folder) if fname.endswith(".ply") and fname.startswith("fused_")]
    print(f"📂 共发现 {len(ply_files)} 个点云待匹配")

    for fused_ply_path in ply_files:
        match_one_query(fused_ply_path, model_dict)

if __name__ == '__main__':
    main()

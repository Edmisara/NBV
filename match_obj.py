import os
import pickle
import numpy as np
import open3d as o3d
import multiprocessing as mp
import cv2

# ==== 可调参数区 ====
voxel_size = 0.015  # 高密度点云推荐 0.01 - 0.02
radius_normal = voxel_size * 2
radius_feature = voxel_size * 5
lambda_weight = 3.0
gamma_weight = 0.2
xi_weight = 0.8  # 长宽比分数因子
baseline_corr = 3000
apply_downsample = True  # 是否强制下采样到统一密度

# ==== 归一化点云 ====
def normalize_pcd(pcd):
    pts = np.asarray(pcd.points)
    pts -= pts.mean(axis=0)
    scale = np.linalg.norm(pts, axis=1).max()
    pts /= scale
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

# ==== 特征提取 ====
def preprocess_point_cloud(pcd):
    if apply_downsample:
        pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd, fpfh

# ==== 轮廓拟合长宽比分数 ====
def compute_aspect_score(pcd):
    points = np.asarray(pcd.points)
    points -= points.mean(axis=0)
    scale = np.linalg.norm(points, axis=1).max()
    points /= scale
    projected = points[:, [0, 2]]  # 投影到XZ平面
    projected += 1
    projected *= 256
    projected = projected.astype(np.int32)
    mask = np.zeros((512, 512), dtype=np.uint8)
    for pt in projected:
        if 0 <= pt[0] < 512 and 0 <= pt[1] < 512:
            mask[pt[1], pt[0]] = 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    w, h = rect[1]
    if h == 0 or w == 0:
        return 0.0
    aspect_ratio = max(w, h) / min(w, h)
    return 1 - abs(aspect_ratio - 1)

# ==== 匹配核心 ====
def match_worker(args):
    query_down_np, query_fpfh_np, query_aspect, fname, model_down_np, model_fpfh_np, model_aspect = args
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
        aspect_score = 1 - abs(query_aspect - model_aspect)
        return fname, fitness, rmse, corr, aspect_score
    except Exception as e:
        return fname, None, None, None, None

# ==== 主流程 ====
def main():
    fused_pcd_path = "D:/NBV/nbv_simulation/results/objects/fused_table.pcd"
    furniture_library_path = "D:/NBV/nbv_simulation/data/furniture_library.pkl"
    model_obj_folder = "D:/NBV/nbv_simulation/data"

    query_pcd = o3d.io.read_point_cloud(fused_pcd_path)
    label = os.path.basename(fused_pcd_path).replace("fused_", "").replace(".pcd", "")
    print(f"[1] 加载查询点云: {fused_pcd_path}\n     → 标签: {label}")

    with open(furniture_library_path, "rb") as f:
        model_dict = pickle.load(f)

    if label not in model_dict:
        print(f"❌ 标签 {label} 不在模型库中")
        return

    query_pcd = normalize_pcd(query_pcd)
    query_down, query_fpfh = preprocess_point_cloud(query_pcd)
    query_down_np = np.asarray(query_down.points)
    query_fpfh_np = query_fpfh.data
    query_aspect = compute_aspect_score(query_down)

    tasks = []
    for fname, pcd_path in model_dict[label]:
        try:
            model_pcd = o3d.io.read_point_cloud(pcd_path)
            model_pcd = normalize_pcd(model_pcd)
            model_down, model_fpfh = preprocess_point_cloud(model_pcd)
            model_down_np = np.asarray(model_down.points)
            model_fpfh_np = model_fpfh.data
            model_aspect = compute_aspect_score(model_down)
            tasks.append((query_down_np, query_fpfh_np, query_aspect, fname, model_down_np, model_fpfh_np, model_aspect))
        except Exception as e:
            print(f"[跳过] {fname} 加载失败：{e}")

    best_model = None
    best_score = -np.inf
    best_model_path = None
    best_fitness = 0

    print("\n🚀 启动多进程匹配...")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(match_worker, tasks)

    for fname, fitness, rmse, corr, aspect_score in results:
        if fitness is None:
            print(f"[跳过] {fname} 匹配失败。")
            continue
        score = fitness - lambda_weight * rmse + gamma_weight * (corr / baseline_corr) + xi_weight * aspect_score
        print(f"[匹配] {fname} → fitness={fitness:.4f}, rmse={rmse:.4f}, corr={corr}, aspect_score={aspect_score:.4f}, score={score:.4f}")
        if score > best_score:
            best_score = score
            best_fitness = fitness
            best_model = fname
            best_model_path = os.path.join(model_obj_folder, fname)

    if best_model:
        print("\n✅ 匹配成功")
        print(f"最相似模型: {best_model}")
        print(f"fitness: {best_fitness:.4f}")
        print(f"score: {best_score:.4f}")
        print(f"OBJ路径: {best_model_path}")
    else:
        print("❌ 未找到匹配模型")

if __name__ == '__main__':
    main()

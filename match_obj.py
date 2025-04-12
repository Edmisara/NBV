import os
import pickle
import numpy as np
import open3d as o3d
import multiprocessing as mp

def normalize_pcd(pcd):
    pts = np.asarray(pcd.points)
    pts -= pts.mean(axis=0)
    scale = np.linalg.norm(pts, axis=1).max()
    pts /= scale
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd

def preprocess_point_cloud(pcd, voxel_size):
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    return pcd, fpfh

def match_worker(args):
    query_down_np, query_fpfh_np, fname, model_down_np, model_fpfh_np, voxel_size = args
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
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))

        result_icp = o3d.pipelines.registration.registration_icp(
            query_down, model_down, voxel_size * 1.5, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())

        fitness = result_icp.fitness
        rmse = result_icp.inlier_rmse
        corr = len(result_icp.correspondence_set)
        return fname, fitness, rmse, corr
    except Exception as e:
        return fname, None, None, None

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

    lambda_weight = 1.0
    gamma_weight = 0.2
    baseline_corr = 1000
    voxel_size = 0.05

    query_pcd = normalize_pcd(query_pcd)
    query_down, query_fpfh = preprocess_point_cloud(query_pcd, voxel_size)
    query_down_np = np.asarray(query_down.points)
    query_fpfh_np = query_fpfh.data

    tasks = []
    for fname, pcd_path in model_dict[label]:
        try:
            model_pcd = o3d.io.read_point_cloud(pcd_path)
            model_pcd = normalize_pcd(model_pcd)
            model_down, model_fpfh = preprocess_point_cloud(model_pcd, voxel_size)
            model_down_np = np.asarray(model_down.points)
            model_fpfh_np = model_fpfh.data
            tasks.append((query_down_np, query_fpfh_np, fname, model_down_np, model_fpfh_np, voxel_size))
        except Exception as e:
            print(f"[跳过] {fname} 加载失败：{e}")

    best_model = None
    best_score = -np.inf
    best_model_path = None
    best_fitness = 0

    print("\n🚀 启动多进程匹配...")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(match_worker, tasks)

    for fname, fitness, rmse, corr in results:
        if fitness is None:
            print(f"[跳过] {fname} 匹配失败。")
            continue
        score = fitness - lambda_weight * rmse + gamma_weight * (corr / baseline_corr)
        print(f"[匹配] {fname} → fitness={fitness:.4f}, rmse={rmse:.4f}, corr={corr}, score={score:.4f}")
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
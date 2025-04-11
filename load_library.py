import os
import open3d as o3d
import pickle
import numpy as np
import pymeshlab
from collections import defaultdict


def parse_label_from_filename(filename):
    return filename.split("_")[0]

def pointcloud_to_dict(pcd):
    return {
        "points": np.asarray(pcd.points),
        "colors": np.asarray(pcd.colors) if pcd.has_colors() else None
    }

def dict_to_pointcloud(pcd_dict):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_dict["points"])
    if pcd_dict["colors"] is not None:
        pcd.colors = o3d.utility.Vector3dVector(pcd_dict["colors"])
    return pcd

def triangulate_with_meshlab(obj_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_path)
    ms.apply_filter("triangulate_all_faces")
    temp_path = obj_path.replace(".obj", "_tri.obj")
    ms.save_current_mesh(temp_path, save_vertex_color=True)
    return temp_path

def load_obj_models_with_labels(folder_path="D:/NBV/nbv_simulation/data", sample_points=2048):
    cache_path = os.path.join(folder_path, "furniture_library.pkl")
    tag_list_path = os.path.join(folder_path, "furniture_labels.txt")
    record_path = os.path.join(folder_path, "loaded_files.txt")

    loaded_files = set()
    if os.path.exists(record_path):
        with open(record_path, "r") as f:
            loaded_files = set(line.strip() for line in f if line.strip())

    if os.path.exists(cache_path):
        print(f"📦 从缓存加载模型库: {cache_path}")
        with open(cache_path, "rb") as f:
            serializable_dict = pickle.load(f)
            model_dict = {
                label: [(fname, dict_to_pointcloud(pcd_dict)) for fname, pcd_dict in models]
                for label, models in serializable_dict.items()
            }
    else:
        model_dict = defaultdict(list)

    updated = False
    current_loaded = set()

    for fname in os.listdir(folder_path):
        if fname.endswith(".obj") and fname not in loaded_files:
            label = parse_label_from_filename(fname)
            full_path = os.path.join(folder_path, fname)

            mesh = o3d.io.read_triangle_mesh(full_path)
            if mesh.is_empty() or len(mesh.triangles) == 0:
                print(f"[提示] 模型 {fname} 为空或未三角化，尝试使用 MeshLab 进行三角化...")
                try:
                    tri_path = triangulate_with_meshlab(full_path)
                    mesh = o3d.io.read_triangle_mesh(tri_path)
                    if mesh.is_empty() or len(mesh.triangles) == 0:
                        print(f"[失败] 三角化失败或加载失败：{tri_path}")
                        continue
                    else:
                        print(f"[成功] 三角化并加载：{tri_path}")
                except Exception as e:
                    print(f"[错误] 三角化过程中出错：{fname} → {e}")
                    continue

            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()

            try:
                pcd = mesh.sample_points_poisson_disk(sample_points)
                model_dict[label].append((fname, pcd))
                current_loaded.add(fname)
                updated = True
                print(f"✅ 已加载模型: {fname}")
            except Exception as e:
                print(f"[错误] 采样失败：{fname} → {e}")

    if updated:
        print(f"💾 正在保存更新后的模型库至: {cache_path}")
        serializable_dict = {
            label: [(fname, pointcloud_to_dict(pcd)) for fname, pcd in models]
            for label, models in model_dict.items()
        }
        with open(cache_path, "wb") as f:
            pickle.dump(serializable_dict, f)

        with open(tag_list_path, "w") as f:
            for label in sorted(model_dict.keys()):
                f.write(label + "\n")
        print(f"🏷️ 标签列表已保存至: {tag_list_path}")

        with open(record_path, "a") as f:
            for fname in current_loaded:
                f.write(fname + "\n")
        print(f"📝 加载记录已更新: {record_path}")
    else:
        print("⚠️ 没有新模型被加载。")

    print(f"📚 当前标签数: {len(model_dict)}, 总模型数: {sum(len(v) for v in model_dict.values())}")
    return model_dict

    
if __name__ == "__main__":
    load_obj_models_with_labels()

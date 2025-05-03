import open3d as o3d
import numpy as np
import os

def create_sphere_obj(points, radius=0.01, sphere_resolution=6, save_path="output.obj"):
    def generate_uv_sphere(res=6):
        vertices = []
        faces = []
        for i in range(res + 1):
            lat = np.pi * i / res
            for j in range(res):
                lon = 2 * np.pi * j / res
                x = np.sin(lat) * np.cos(lon)
                y = np.sin(lat) * np.sin(lon)
                z = np.cos(lat)
                vertices.append((x, y, z))
        for i in range(res):
            for j in range(res):
                p0 = i * res + j
                p1 = p0 + 1 if (j + 1) < res else i * res
                p2 = p0 + res
                p3 = p1 + res
                if i != res - 1:
                    faces.append((p0, p2, p3))
                    faces.append((p0, p3, p1))
        return np.array(vertices), np.array(faces)

    v_template, f_template = generate_uv_sphere(sphere_resolution)
    obj_lines = []
    vert_offset = 1

    for p in points:
        v_scaled = v_template * radius + p
        for v in v_scaled:
            obj_lines.append(f"v {v[0]} {v[1]} {v[2]}")
        for f in f_template:
            f_str = " ".join(str(idx + vert_offset) for idx in f)
            obj_lines.append(f"f {f_str}")
        vert_offset += len(v_template)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write("\n".join(obj_lines))

    print(f"✅ Saved OBJ with {len(points)} spheres to:\n{save_path}")

# ====== 主程序入口 ======
if __name__ == "__main__":
    ply_path = "D:/NBV/nbv_simulation/results/model_full_transformed.ply"
    obj_path = "D:/NBV/nbv_simulation/results/model_full_transformed_spheres.obj"

    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    print(f"原始点数: {points.shape[0]}")

    # ✅ 坐标转换：Open3D → Blender
    points = points[:, [0, 2, 1]]  # X, Z, Y
    points[:, 2] *= -1             # 反转 Z（Y 轴翻转）

    create_sphere_obj(points, radius=0.01, sphere_resolution=6, save_path=obj_path)

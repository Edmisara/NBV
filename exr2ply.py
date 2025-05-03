import numpy as np
import json
import os
from pathlib import Path

def convert_npy_to_ply(base_path, output_name="cloud.ply"):
    base_path = Path(base_path)
    depth_path = base_path / "depth_meter.npy"
    camera_path = base_path / "camera.json"
    ply_output_path = base_path / output_name

    if not depth_path.exists() or not camera_path.exists():
        print(f"❌ 缺少 depth 或 camera：{base_path}")
        return

    depth = np.load(depth_path)

    with open(camera_path, "r") as f:
        cam = json.load(f)

    fx, fy = cam["intrinsic"]["fx"], cam["intrinsic"]["fy"]
    cx, cy = cam["intrinsic"]["cx"], cam["intrinsic"]["cy"]
    width, height = cam["intrinsic"]["width"], cam["intrinsic"]["height"]
    T_world2cam = np.array(cam["extrinsic"])
    T_cam2world = np.linalg.inv(T_world2cam)

    points = []

    for v in range(height):
        for u in range(width):
            z = depth[v, u]
            if z == 0:
                continue
            x = (u - cx) / fx
            y = -(v - cy) / fy
            ray = np.array([x, y, -1.0])
            ray /= np.linalg.norm(ray)
            point_cam = ray * z
            point_world = T_cam2world @ np.append(point_cam, 1.0)
            points.append(point_world[:3])

    with open(ply_output_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")

        for pt in points:
            f.write(f"{pt[0]} {pt[1]} {pt[2]}\n")

    print(f"✅ PLY 点云已保存：{ply_output_path}（共 {len(points)} 点）")


# === 示例主函数 ===
def main():
    convert_npy_to_ply("D:/NBV/nbv_simulation/results/CamerasCatch/cam_0")

    base_dir = Path("D:/NBV/nbv_simulation/results/CamerasCatch")
    for subdir in base_dir.iterdir():
        if subdir.is_dir():
            convert_npy_to_ply(subdir)

if __name__ == "__main__":
    main()

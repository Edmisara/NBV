import numpy as np
import trimesh
import pyrender
import cv2
from PIL import Image
import os
from skimage.metrics import structural_similarity as ssim

# === 参数设置 ===
mesh_path = "D:/NBV/nbv_simulation/data/table_001.obj"
intrinsic_path = "D:/NBV/nbv_simulation/results/intrinsic_matrix.npy"
extrinsic_path = "D:/NBV/nbv_simulation/results/extrinsic_matrix.npy"
target_image_path = "D:/NBV/nbv_simulation/results/objects/fused_table.png"
output_dir = "D:/NBV/nbv_simulation/results/objects/pose_match_search"
os.makedirs(output_dir, exist_ok=True)

# === 加载目标轮廓图（透明背景 alpha）===
target_image = Image.open(target_image_path).convert("RGBA")
alpha = np.array(target_image)[:, :, 3]
target_mask = (alpha > 10).astype(np.uint8) * 255
height, width = target_mask.shape

# === 相机参数 ===
intrinsic = np.load(intrinsic_path)
extrinsic = np.load(extrinsic_path)
T_cw = np.linalg.inv(extrinsic)
fx, fy = intrinsic[0, 0], intrinsic[1, 1]
cx, cy = intrinsic[0, 2], intrinsic[1, 2]
camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.05, zfar=5.0)

# === 加载并标准化 mesh ===
o3d_mesh = trimesh.load(mesh_path, process=False)
bbox = o3d_mesh.bounds
scale_base = 1.0 / np.linalg.norm(bbox[1] - bbox[0])
center = o3d_mesh.centroid
o3d_mesh.apply_scale(scale_base)
o3d_mesh.apply_translation(-center)
R_x = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
o3d_mesh.apply_transform(R_x)

# === 搜索参数 ===
scale_factors = np.linspace(0.9, 1.1, 3)
shifts = np.linspace(-0.2, 0.2, 5)  # 搜索平移
z_base = 1.5

# === 初始化搜索 ===
best_score = -np.inf
best_params = None
best_rgb = None
best_silhouette = None

renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)

# === 开始搜索 ===
for s in scale_factors:
    for dx in shifts:
        for dy in shifts:
            for dz in shifts:
                mesh = o3d_mesh.copy()
                mesh.apply_scale(s)
                mesh.apply_translation([dx, dy, z_base + dz])
                
                scene = pyrender.Scene(bg_color=[255, 255, 255, 0], ambient_light=[1.0, 1.0, 1.0])
                pm = pyrender.Mesh.from_trimesh(mesh, smooth=False)
                scene.add(pm)
                scene.add(camera, pose=T_cw)
                scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=T_cw)

                color, _ = renderer.render(scene)
                gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
                silhouette = (gray > 10).astype(np.uint8) * 255

                score = ssim(silhouette, target_mask)
                if score > best_score:
                    best_score = score
                    best_params = (s, dx, dy, dz)
                    best_rgb = color.copy()
                    best_silhouette = silhouette.copy()

renderer.delete()

# === 保存最佳结果 ===
print("✅ 搜索完成！")
print("🎯 最佳匹配参数：")
print(f"Scale: {best_params[0]:.4f}")
print(f"Translation: dx={best_params[1]:.4f}, dy={best_params[2]:.4f}, dz={best_params[3]:.4f}")
print(f"📈 匹配 SSIM 得分: {best_score:.4f}")

cv2.imwrite(os.path.join(output_dir, "best_match_rgb.png"), cv2.cvtColor(best_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, "best_match_silhouette.png"), best_silhouette)

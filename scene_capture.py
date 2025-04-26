import blenderproc as bproc
import bpy
import numpy as np
import os
import json
from PIL import Image

# === 初始化场景 ===
bproc.init()

# === 加载 .blend 场景模型（不依赖相机了）===
bproc.loader.load_blend("D:/NBV/nbv_simulation/view.blend")

# === 输出目录 ===
output_dir = "D:/NBV/nbv_simulation/results"
os.makedirs(output_dir, exist_ok=True)

# === 相机分辨率 ===
width, height = 1920, 1080
bproc.camera.set_resolution(width, height)

# ✅ 添加摄像机
bpy.ops.object.camera_add(location=(60, -60, 40), rotation=(1.3, 0, 0.7))
bpy.context.view_layer.update()  # ⬅️ 刷新 matrix_world
cam = bpy.context.active_object
bpy.context.scene.camera = cam
cam.data.lens = 18  # mm

# ✅ 获取真实姿态（世界到相机）
T_world2cam = np.linalg.inv(np.array(cam.matrix_world))
T_cam2world = np.array(cam.matrix_world)
bproc.camera.add_camera_pose(T_cam2world)

# === 设置相机内参 ===
bproc.camera.set_intrinsics_from_blender_params(
    lens=18.0,
    image_width=width,
    image_height=height,
    lens_unit="MILLIMETERS"
)

# ✅ 启用真实距离渲染（而非 Z-buffer 深度）
bproc.renderer.enable_distance_output(activate_antialiasing=False)  # 渲染单位为“米”的真实距离图

# === 渲染设置 ===
bproc.renderer.set_output_format(enable_transparency=False)
bproc.renderer.set_light_bounces(1, 1, 1, 1)

# === 渲染并获取图像数据 ===
data = bproc.renderer.render()

# === 保存 RGB 图像 ===
rgb_img = (data["colors"][0] * 255).astype(np.uint8)
Image.fromarray(rgb_img).save(os.path.join(output_dir, "rgb.png"))

depth = data["distance"][0]  # 原始深度图（米）

# ✅ 设置合理的最小最大距离
min_depth = 1  # 最小有效距离
max_depth = 6000.0  # 最大有效距离

# ✅ 筛选出合理范围内的深度，否则设为0
filtered_depth = np.where((depth >= min_depth) & (depth <= max_depth), depth, 0)

# ✅ 保存筛选后的深度图
np.save(os.path.join(output_dir, "depth_meter.npy"), filtered_depth)

# === 保存相机内外参 ===
K = bproc.camera.get_intrinsics_as_K_matrix()
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

cam_info = {
    "intrinsic": {
        "fx": float(fx), "fy": float(fy),
        "cx": float(cx), "cy": float(cy),
        "width": width, "height": height
    },
    "extrinsic": T_world2cam.tolist()
}

with open(os.path.join(output_dir, "camera.json"), "w") as f:
    json.dump(cam_info, f, indent=4)

print("✅ 渲染完成！RGB 图、真实深度图和相机参数已保存到：", output_dir)

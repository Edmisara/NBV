import bpy
import mathutils

# === 先清空场景（可选）
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# === 你的两个点云文件路径（请改成你自己的）
cloud_clean_path = r"D:/NBV/nbv_simulation/results/cloud_clean.ply"
model_full_path = r"D:/NBV/nbv_simulation/results/model_full_transformed.ply"

# === 导入 Cloud_Clean
bpy.ops.import_mesh.ply(filepath=cloud_clean_path)
cloud_clean_obj = bpy.context.selected_objects[0]
cloud_clean_obj.name = "Cloud_Clean"

# === 导入 Model_Full
bpy.ops.import_mesh.ply(filepath=model_full_path)
model_full_obj = bpy.context.selected_objects[0]
model_full_obj.name = "Model_Full"

# === 相机朝向补偿（基于原相机旋转）
cam_rotation_euler = mathutils.Euler((1.3, 0.0, 0.7), 'XYZ')
cam_rotation_matrix = cam_rotation_euler.to_matrix().to_4x4()

# === 对两个点云应用旋转
for obj in [cloud_clean_obj, model_full_obj]:
    obj.matrix_world = cam_rotation_matrix @ obj.matrix_world

print("[LOG] Cloud_Clean 和 Model_Full 已按相机朝向旋转修正。")

# === 添加一个新相机
bpy.ops.object.camera_add(location=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0))
cam = bpy.context.active_object
bpy.context.scene.camera = cam
cam.data.lens = 18  # 焦距设成你原来的18mm

print("[LOG] 相机已放在原点，朝向标准前方，焦距18mm。")

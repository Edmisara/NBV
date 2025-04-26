
import bpy
import os

# === 清空场景 ===
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# === 路径设置 ===
base_path = r"D:\NBV\nbv_simulation\logs"
chair_path = os.path.join(base_path, "chair.obj")
table_path = os.path.join(base_path, "table.obj")

# === 导入模型 ===
bpy.ops.import_scene.obj(filepath=chair_path)
bpy.ops.import_scene.obj(filepath=table_path)

# === 设置位置 ===
for obj in bpy.context.selected_objects:
    if "chair" in obj.name.lower():
        obj.location = (0.0, 0.0, 0.0)
    elif "table" in obj.name.lower():
        obj.location = (0.0, -30.0, 0.0)

# === 添加摄像机 ===
bpy.ops.object.camera_add(location=(60, -60, 40), rotation=(1.3, 0, 0.7))
cam = bpy.context.active_object
bpy.context.scene.camera = cam
cam.data.lens = 18

# === 添加光源 ===
bpy.ops.object.light_add(type='SUN', location=(5, -5, 5))

# === 材质赋予函数 ===
def assign_material(obj, color=(1, 1, 1, 1), metallic=0.0, roughness=0.5):
    mat = bpy.data.materials.new(name=f"{obj.name}_Material")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

# === 材质分配 ===
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        if "chair" in obj.name.lower():
            assign_material(obj, color=(0.2, 0.5, 0.8, 1), metallic=0.1, roughness=0.3)
        elif "table" in obj.name.lower():
            assign_material(obj, color=(0.6, 0.4, 0.2, 1), metallic=0.0, roughness=0.6)

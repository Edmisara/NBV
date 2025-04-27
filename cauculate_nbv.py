import bpy
import math
from mathutils import Vector, Matrix

# ================== 参数设置 ==================
num_candidates_initial = 16
fine_sampling_count = 8
fine_search_angle = math.radians(45)
binary_search_steps = 3
depth_limit = 100.0  # 可见深度上限

target_obj_name = "model_full_transformed"

# ================== 工具函数 ==================

def get_object_points(obj_name):
    obj = bpy.data.objects[obj_name]
    points = []
    for v in obj.data.vertices:
        world_coord = obj.matrix_world @ v.co
        points.append(world_coord)
    return points

def get_center(points):
    center = Vector((0.0, 0.0, 0.0))
    for p in points:
        center += p
    return center / len(points)

def generate_candidate_views(center, radius, height, num_views, center_theta):
    views = []
    start_angle = center_theta - math.radians(22.5)
    for i in range(num_views):
        theta = start_angle + i * (math.radians(45) / (num_views - 1))
        x = center.x + radius * math.cos(theta)
        y = center.y + radius * math.sin(theta)
        z = center.z + height
        location = Vector((x, y, z))
        direction = (center - location).normalized()
        rot_quat = direction_to_quaternion(direction)
        views.append((location, rot_quat, theta))
    return views

def generate_initial_views(center, radius, height, num_views):
    views = []
    for i in range(num_views):
        theta = 2 * math.pi * i / num_views
        x = center.x + radius * math.cos(theta)
        y = center.y + radius * math.sin(theta)
        z = center.z + height
        location = Vector((x, y, z))
        direction = (center - location).normalized()
        rot_quat = direction_to_quaternion(direction)
        views.append((location, rot_quat, theta))
    return views

def direction_to_quaternion(direction):
    up = Vector((0, 0, 1))
    right = up.cross(direction).normalized()
    up = direction.cross(right)

    rot = Matrix((
        right,
        up,
        direction
    )).transposed()

    return rot.to_quaternion()

def is_visible(scene, depsgraph, origin, target):
    direction = target - origin
    distance_target = direction.length
    direction_norm = direction.normalized()

    hit, loc, normal, idx, obj, matrix = scene.ray_cast(depsgraph, origin, direction_norm, distance=distance_target)
    if not hit:
        return True  # 直接打到目标，ok
    
    distance_hit = (loc - origin).length
    if distance_hit >= distance_target * 0.99:
        return True  # 没有中途撞到障碍物，允许微小误差
    else:
        return False  # 中途撞到，判定为被遮挡

def get_visible_points(scene, depsgraph, points_world, cam_location):
    visible_idx = []
    for idx, pt in enumerate(points_world):
        if (pt - cam_location).length > depth_limit:
            continue
        if is_visible(scene, depsgraph, cam_location, pt):
            visible_idx.append(idx)
    return set(visible_idx)

# ================== 主流程 ==================

scene = bpy.context.scene
depsgraph = bpy.context.evaluated_depsgraph_get()
orig_cam = scene.camera

# 原相机内参
lens = orig_cam.data.lens
sensor_width = orig_cam.data.sensor_width
sensor_height = orig_cam.data.sensor_height
resolution_x = scene.render.resolution_x
resolution_y = scene.render.resolution_y

fx = lens / sensor_width * resolution_x
fy = lens / sensor_height * resolution_y
cx = resolution_x / 2
cy = resolution_y / 2

print("\n=== Initial Camera Parameters ===")
print(f"fx: {fx}")
print(f"fy: {fy}")
print(f"cx: {cx}")
print(f"cy: {cy}")
print(f"width: {resolution_x}")
print(f"height: {resolution_y}")
print(f"Camera Location: {tuple(orig_cam.location)}")
print(f"Camera Rotation (Euler XYZ): {tuple(orig_cam.rotation_euler)}\n")

# 点云
points_world = get_object_points(target_obj_name)
center = get_center(points_world)

print(f"Total points in model_full_transformed: {len(points_world)}")

# 初始可见点
initial_visible_set = get_visible_points(scene, depsgraph, points_world, orig_cam.location)
print(f"Initial visible points: {len(initial_visible_set)}\n")

# 初步撒点
offset = orig_cam.location - center
radius = math.sqrt(offset.x**2 + offset.y**2)
height = offset.z

candidates = generate_initial_views(center, radius, height, num_candidates_initial)

# ================== 三次二分搜索 ==================

current_best_theta = None

for step in range(binary_search_steps):
    print(f"--- Binary Search Step {step+1} ---")
    if step == 0:
        views = candidates
    else:
        views = generate_candidate_views(center, radius, height, fine_sampling_count, current_best_theta)

    best_gain = -1
    best_view = None

    for loc, rot, theta in views:
        visible_set = get_visible_points(scene, depsgraph, points_world, loc)
        newly_visible = visible_set - initial_visible_set
        gain = len(newly_visible)
        print(f"Candidate at theta {math.degrees(theta):.2f}°: gain {gain}")

        if gain > best_gain:
            best_gain = gain
            best_view = (loc, rot, visible_set, theta)

    current_best_loc, current_best_rot, current_best_visible_set, current_best_theta = best_view
    print(f"Best gain in this step: {best_gain}\n")

# ================== 创建 NBV 相机 ==================

bpy.ops.object.camera_add(location=current_best_loc)
new_cam = bpy.context.active_object
new_cam.data.lens = lens
new_cam.data.sensor_width = sensor_width
new_cam.data.sensor_height = sensor_height
scene.camera = new_cam
new_cam.rotation_mode = 'QUATERNION'
new_cam.rotation_quaternion = current_best_rot

new_total_visible = len(initial_visible_set.union(current_best_visible_set))
newly_visible_count = len(current_best_visible_set - initial_visible_set)

print("\n=== NBV Final Result ===")
print(f"Best NBV Location: {tuple(current_best_loc)}")
print(f"Best NBV Rotation (Quaternion): {tuple(current_best_rot)}")
print(f"Newly visible points from NBV: {newly_visible_count}")
print(f"Total visible points after NBV: {new_total_visible}")
print(f"Information Gain: {newly_visible_count/len(points_world)*100:.2f}%\n")

import open3d as o3d
import numpy as np
import copy
import os
from PIL import Image
from create_scene import create_scene
from camera_utils import save_image_from_camera


def add_camera_button(vis, camera_position, camera_euler, scene):
    # 按钮回调函数
    def button_callback(vis):
        # 调用 save_image_from_camera 函数
        save_image_from_camera(vis, path="D:/NBV/nbv_simulation/results")
        return False  # 关闭按钮的操作

    # 创建按钮
    vis.register_key_callback(ord("S"), button_callback)  # 按下"S"键进行拍照

def render_scene(geometry_list, camera_position, camera_euler):
    # 使用 VisualizerWithKeyCallback
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="3D Scene with Camera", visible=True)
    
    # 添加场景几何体
    for g in geometry_list:
        vis.add_geometry(copy.deepcopy(g))  # 使用 copy.deepcopy 确保不修改原始几何体
    
    # 设置相机视角为相机的位置
    ctr = vis.get_view_control()

    # 设置相机视角的方向
    front_direction = camera_position / np.linalg.norm(camera_position)  # 设置相机朝向
    ctr.set_front(front_direction)  # 设置相机前方向
    ctr.set_lookat([0, 0, 0])  # 让相机朝向场景的原点
    ctr.set_up([0, 0, 1])  # 设置相机的“上”方向

    # 按钮控制拍摄
    add_camera_button(vis, camera_position, camera_euler, geometry_list)

    vis.poll_events()
    vis.update_renderer()

    # 保持窗口打开直到用户关闭
    vis.run()


# 创建场景和相机
scene = create_scene()
camera_position = np.array([0, -40, 40])  # 离地40高度
camera_euler = np.radians([45, 0, 0])     # pitch, yaw, roll

# 渲染场景并添加相机
render_scene(scene, camera_position, camera_euler)
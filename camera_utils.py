import os
import numpy as np
import open3d as o3d
from PIL import Image
import json

def save_image_from_camera(vis, camera_position, camera_euler, path="D:\\NBV\\nbv_simulation\\results"):
    # 确保保存路径存在，如果不存在则创建
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"目录 {path} 创建成功。")
    else:
        print(f"目录 {path} 已存在。")

    # 获取视图控制器
    ctr = vis.get_view_control()
    
    # 获取当前的相机内参
    intrinsic = ctr.convert_to_pinhole_camera_parameters().intrinsic

    # 构造相机外参
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz(camera_euler)
    extrinsic[:3, 3] = camera_position

    # 设置相机参数
    camera_params = o3d.camera.PinholeCameraParameters()
    camera_params.intrinsic = intrinsic
    camera_params.extrinsic = extrinsic

    intrinsic_matrix = np.array([
        [intrinsic.get_focal_length()[0], 0, intrinsic.get_principal_point()[0]],
        [0, intrinsic.get_focal_length()[1], intrinsic.get_principal_point()[1]],
        [0, 0, 1]
    ])

    # 保存外参矩阵
    extrinsic_matrix = np.array(extrinsic)  # extrinsic 可能是 4x4 矩阵

    # 保存为 .npy 文件
    intrinsic_matrix_path = os.path.join(path, "intrinsic_matrix.npy")
    extrinsic_matrix_path = os.path.join(path, "extrinsic_matrix.npy")

    np.save(intrinsic_matrix_path, intrinsic_matrix)
    np.save(extrinsic_matrix_path, extrinsic_matrix)

    # 捕获 RGB 图像
    rgb_image = vis.capture_screen_float_buffer(do_render=True)
    rgb_image_np = np.asarray(rgb_image) * 255  # 转换为 NumPy 数组并缩放到 0-255 范围
    rgb_image_pil = Image.fromarray(rgb_image_np.astype(np.uint8))
    rgb_image_path = os.path.join(path, "camera_view.png")
    rgb_image_pil.save(rgb_image_path)
    print(f"RGB 图像已保存为 {rgb_image_path}")

    # 捕获深度图像
    depth_image = vis.capture_depth_float_buffer(do_render=True)
    depth_image_np = np.asarray(depth_image)
    depth_image_pil = Image.fromarray(depth_image_np.astype(np.uint8))
    depth_image_path = os.path.join(path, "depth_image.png")
    depth_image_pil.save(depth_image_path)
    print(f"深度图像已保存为 {depth_image_path}")

    # 创建 RGBD 图像
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb_image_np.astype(np.uint8)),
        o3d.geometry.Image(depth_image_np.astype(np.uint8)),
        depth_scale=1000.0,  # 根据实际深度图像的单位进行调整
        depth_trunc=3.0,
        convert_rgb_to_intensity=False
    )

    # 从 RGBD 图像创建点云
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(
        width=rgb_image_np.shape[1],
        height=rgb_image_np.shape[0],
        fx=intrinsic.get_focal_length()[0],
        fy=intrinsic.get_focal_length()[1],
        cx=intrinsic.get_principal_point()[0],
        cy=intrinsic.get_principal_point()[1]
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

    # 点云坐标系转换（可选，根据需要进行调整）
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # 保存点云
    pcd_filename = os.path.join(path, "point_cloud.pcd")
    o3d.io.write_point_cloud(pcd_filename, pcd)
    print(f"点云已保存为 {pcd_filename}")
    return 
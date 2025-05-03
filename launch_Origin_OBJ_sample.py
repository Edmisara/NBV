import subprocess
import os
import sys

# === 1. 设置你的 BlenderProc 脚本路径 ===
scene_capture_script = os.path.abspath("D://NBV//nbv_simulation//Origin_OBJ_sample.py")

# === 2. 可选：检查脚本是否存在 ===
if not os.path.exists(scene_capture_script):
    print(f"❌ 找不到脚本文件：{scene_capture_script}")
    sys.exit(1)

# === 3. 构建 BlenderProc 命令 ===
cmd = [
    "blenderproc",
    "run",
    scene_capture_script
]

print("🚀 启动 BlenderProc:")
print("   脚本:", scene_capture_script)
print("   将自动使用 BlenderProc 内部配置的 Blender（可能自动下载）")
print("💡 当前不会干扰你本地的 Blender 2.93，完全隔离\n")

# === 4. 调用 BlenderProc（自动使用缓存或下载）===
try:
    subprocess.run(cmd, check=True)
except subprocess.CalledProcessError as e:
    print(f"❌ BlenderProc 执行失败，错误码 {e.returncode}")
    sys.exit(e.returncode)
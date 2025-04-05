#!/bin/bash

# ==== 1. 项目名称 ====
PROJECT_NAME="nbv_simulation"
ENV_NAME="nbv-env"
PYTHON_VERSION="3.10"

# ==== 2. 创建项目文件夹结构 ====
mkdir -p $PROJECT_NAME/{src,data,logs,notebooks,configs,results}
cd $PROJECT_NAME

# ==== 3. 创建 Conda 环境 ====
echo "Creating conda environment '$ENV_NAME'..."
conda create -y -n $ENV_NAME python=$PYTHON_VERSION
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# ==== 4. 安装常用依赖 ====
echo "Installing dependencies..."
conda install -y numpy scipy matplotlib ipykernel
conda install -y -c open3d-admin open3d
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 可选：如果你使用 NBV 模型，比如 SurfEmb、VPFH-Net 等，可在此处安装：
# pip install -r requirements.txt

# ==== 5. 创建初始化文件 ====
echo "Creating main.py and README.md..."
cat <<EOT >> src/main.py
# Entry point for NBV simulation
if __name__ == "__main__":
    print("NBV Simulation Environment Ready!")
EOT

cat <<EOT >> README.md
# NBV Simulation Project

This project sets up a simulation pipeline for Next-Best-View (NBV) research using CLIP + RGB-D strategies.
EOT

echo "✅ Project '$PROJECT_NAME' initialized successfully!"

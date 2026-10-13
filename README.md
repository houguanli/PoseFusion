# PoseFusion

PoseFusion: Fusing neural implicit surfaces for multi-view reconstruction from multi-pose captures

[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0010448526000710) · [Project page](https://raining00.github.io/PoseFusion-page/)

本分支提供整理后的 Voxurf base 代码及配套带背景数据集。

- [代码、环境安装与运行步骤](code/README.md)
- [数据集说明](dataset/README.md)
- [case 与配置索引](code/configs/dataset.json)
- [历史 NeuS base 说明](README_NEUS.md)

## 获取完整文件

图像、GT mesh、NeuS 相机及预训练网络权重通过 Git LFS 保存。先安装 Git 和 Git LFS，再执行：

```bash
git lfs install
git clone --branch release --single-branch https://github.com/houguanli/PoseFusion.git
cd PoseFusion
git lfs pull
cd code
```

本仓库公开，浏览及使用上述命令下载完整代码、数据集和模型权重均无需登录 GitHub。Git LFS 用于取得实际二进制文件；仅下载默认源码 ZIP 可能得到 LFS 指针文件。获取实际二进制文件的方法见 [Git LFS 协作说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/collaboration-with-git-large-file-storage)。

## 发布内容

```text
code/       # Voxurf 重建、配准、fusion、环境文件、配置及推理依赖
dataset/    # 18 case / 54 pose / 2139 RGB 图像及 mask，虚拟数据附 GT
```

数据版本为 `background_inputs_20260929`：11 个虚拟 case、1350 张图像；7 个真实 case、789 张图像。相机采用 NeuS/IDR format。code 内含 108 份重建配置、54 份配准配置，以及 eLoFTR 和 ResNet18 推理权重。

数据读取与采样 fusion 已完成检查；本分支不包含新训练 checkpoint、预测配准结果或临时测试输出。重建、配准、fusion 及 GT 评估按 [code/README.md](code/README.md) 执行。

原有 NeuS 源文件沿用 master，作为历史实现参考；本版统一数据的运行入口位于 `code/pipeline.py`。运行时保持 `code` 与 `dataset` 同级。本仓库以 `honkai_neus` 的 `release` 文件快照为初始版本。

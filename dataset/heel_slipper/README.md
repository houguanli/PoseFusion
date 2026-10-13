# Heel Slipper (heel_slipper)

数据类型：虚拟渲染。状态：complete_background。

主输入 `poseN/image/NNN.png` 是带真实场景背景的 RGB PNG；`mask/NNN.png` 是主体 mask。相机使用 NeuS 的 `cameras_sphere.npz`。

| Pose | 图像 / mask 数 | 图像格式 |
|---|---:|---|
| 1 | 40 | 800x800 RGB |
| 2 | 40 | 800x800 RGB |
| 3 | 40 | 800x800 RGB |

保持此前新版数据的主体、相机和帧序；仅恢复背景主图。

GT：gt/meshes/poseN.ply and gt/transforms.json; source geometry coordinates, no estimated alignment

相机：列向量投影 `p ~ world_mat_i @ scale_mat_i @ X`，取前三维再除以第三分量。帧 `NNN.png` 对应 NPZ 后缀 `_i` 的索引 i。

逐帧背景来源和 SHA-256 记录于 `case_manifest.json`。

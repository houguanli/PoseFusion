# Seal (seal)

数据类型：虚拟渲染。状态：complete_background。

主输入 `poseN/image/NNN.png` 是带真实场景背景的 RGB PNG；`mask/NNN.png` 是主体 mask。相机使用 NeuS 的 `cameras_sphere.npz`。

| Pose | 图像 / mask 数 | 图像格式 |
|---|---:|---|
| 1 | 40 | 800x800 RGB |
| 2 | 40 | 800x800 RGB |
| 3 | 40 | 800x800 RGB |

New 40 frames/pose textured grounded scene. All geometry, GT and images use this new scene. Raw and canonical-origin RT both retained.

GT：gt/meshes/poseN.ply and gt/transforms.json; source geometry coordinates, no estimated alignment

相机：列向量投影 `p ~ world_mat_i @ scale_mat_i @ X`，取前三维再除以第三分量。帧 `NNN.png` 对应 NPZ 后缀 `_i` 的索引 i。

逐帧背景来源和 SHA-256 记录于 `case_manifest.json`。

海豹 pose1 与 pose2 原始局部原点不同。`transforms.json` 使用 pose2 共同局部坐标系计算相对 R/t，保留原始 Blender 矩阵与原点修正。未对结果做拟合，也未更改渲染的世界几何。

原先 image_wb 中部分俯视图的地面接近纯黑，本版背景从已确认的 seal_background.blend 单独按原 camera_data.json 重渲染，原 mask、相机、GT mesh 与 R/t 未改。论文中的失败 case 标签仅用于关联，本版没有重跑失败结论。

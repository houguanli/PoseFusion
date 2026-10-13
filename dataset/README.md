# PoseFusion 带背景数据集

本目录仅包含输入数据、虚拟数据 GT、来源记录与校验脚本。历史预测 mesh、配准结果、去背景版本和旧版备份位于 `F:/HGL_WORK_PASSED/posefusion/dataset_reorganization_20260929/archive`，不混入本版数据。

本版包含 18 个带背景 case：11 个虚拟、7 个真实。所有主 image 目录均为实际背景图像。

## 数据清单

| Case | 类型 | Pose1 / Pose2 / Pose3 | 状态 |
|---|---|---:|---|
| Bunny (`bunny`) | 虚拟 | 40 / 40 / 40 | complete_background |
| Book (`book`) | 虚拟 | 45 / 45 / 45 | complete_background |
| Nikon (`nikon`) | 虚拟 | 40 / 40 / 40 | complete_background |
| Heel Slipper (`heel_slipper`) | 虚拟 | 40 / 40 / 40 | complete_background |
| Figurine A (`figurine_a`) | 虚拟 | 45 / 45 / 45 | complete_background |
| Tank (`tank`) | 虚拟 | 40 / 40 / 40 | complete_background |
| Monster (`monster`) | 虚拟 | 40 / 40 / 40 | complete_background |
| Figurine B (`figurine_b`) | 虚拟 | 40 / 40 / 40 | complete_background |
| Rubik’s Cube (`rubik`) | 虚拟 | 40 / 40 / 40 | complete_background |
| Lego (`lego`) | 虚拟 | 40 / 40 / 40 | complete_background |
| Seal (`seal`) | 虚拟 | 40 / 40 / 40 | complete_background |
| Figurine C (`figurine_c`) | 真实 | 32 / 32 / 32 | complete_background |
| Figurine D (`figurine_d`) | 真实 | 33 / 32 / 33 | complete_background |
| Figurine E (`figurine_e`) | 真实 | 32 / 33 / 32 | complete_background |
| Figurine F / Keli (`figurine_f`) | 真实 | 55 / 36 / 35 | complete_background |
| Joystick (`joystick`) | 真实 | 32 / 32 / 32 | complete_background |
| Airplane (`airplane`) | 真实 | 32 / 32 / 32 | complete_background |
| Guardian (`guardian`) | 真实 | 60 / 60 / 60 | complete_background |

数量以 `dataset_manifest.json` 为准：

- 虚拟：11 个完整 case，1350 张图像。
- 真实：7 个完整 case，789 张图像。

## 输入格式

- 每个 case 一个子目录，每个 pose 分开保存。主输入为 `poseN/image/000.png` 等 RGB PNG，包含实际渲染场景或拍摄背景；没有另设 `image_wb` 让使用者自行切换。
- `poseN/mask/000.png` 为对应主体 mask，尺寸与主图一致。虚拟 mask 延用同相机的原前景轮廓，不把地面、阴影或倒影算成主体；它不是新渲染背景的语义分割标签。
- 虚拟通常为 800×800；Monster 为 1200×800。真实 case 通常为 1024×1024；Keli 为 1920×1080。具体尺寸、SHA-256 和逐帧来源见各 case 的 manifest。
- `poseN/cameras_sphere.npz` 为 NeuS/IDR 相机格式。图像编号 i 对应 `world_mat_i`、`scale_mat_i`，以及源文件存在时的逆矩阵和 `camera_mat_i`。列向量投影：`p ~ world_mat_i @ scale_mat_i @ X`，取前三维并除以第三分量。
- 虚拟附带原 `camera_data.json`；其中 `_K` 为内参，`_M` 为 OpenCV camera-to-world。不得再按 Blender 轴约定直接使用 `_M`。

## GT 与版本

- 11 个虚拟 case 的 `gt/meshes/poseN.ply` 与 `gt/transforms.json` 均基于 E 盘新版源场景。相对变换定义为列向量 `x_target = T_source_to_target @ x_source`。不把预测 mesh 或估计变换当作 GT。
- 10 个常规虚拟 case 按已保存相机逐帧补渲染，主体原始世界变换保持不变；海豹使用已完成的新版 40×3 张渲染、对应 mesh 与 GT。旧版海豹 35×3 张不再混入。
- 海豹原 pose1 的局部原点与 pose2/3 不同。配准 GT 用 pose2 统一局部坐标系，`gt/transforms.json` 同时保留原始 Blender 世界矩阵和局部原点修正。原点残差约 3.1e-8；没有依据预测结果拟合 GT。
- Tank / Monster 原场景的相机采样辅助球 Sphere 在视口隐藏但仍参与渲染；补渲染仅关闭辅助球渲染，不移动主体或相机。缺失贴图的旧绝对路径按同一源目录现有资源重新绑定。
- Keli 126 张直接复制自用户补充的 reg/keli_full_wm/keli_iphone_poseN/images，前景像素和相机与旧版完全一致。Guardian 同样直接恢复原带背景图；其余真实图像由未抠图、已标定原图按相机投影矩阵推导的整数窗口裁剪，并逐帧核对原前景像素；未生成或拼接假背景。
- Guardian 保留原 Gardian 60/60/60 帧，背景来自 raw_video/realworld_datasets_2/sliced_images_sliced，逐帧前景像素和相机完全对应。没有采用另一套 Guardian 32 张/pose 的采集数据。
- 真实数据没有独立测量的配准 RT/mesh GT；本次恢复背景也没有重跑论文方法或重建训练。

## 校验与溯源

`provenance/` 保存恢复/渲染脚本及来源核对记录；脚本中的外部路径是此次整理的来源记录，移机时需修改。`file_manifest.json` 记录本版文件哈希。

安装 NumPy 与 Pillow 后运行 `python validate_dataset.py`，验证 RGB 背景内容、图像/mask 尺寸、帧数与相机数、相机 JSON 一致性、RT 组成/正交性及文件哈希。`validation_report.json` 记录全部 case 的校验结果。

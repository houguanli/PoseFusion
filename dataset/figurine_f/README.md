# Figurine F / Keli (figurine_f)

真实拍摄数据，三个 pose 的 image 均为带实际背景的 RGB PNG。配套 mask 与 NeuS cameras_sphere.npz 保持原帧序。

| Pose | 图像 / mask 数 | 格式 |
|---|---:|---|
| 1 | 55 | 1920x1080 RGB |
| 2 | 36 | 1920x1080 RGB |
| 3 | 35 | 1920x1080 RGB |

没有独立测量的配准 RT / mesh GT；未将预测结果当作 GT。

原图直接复制自 F:/HGL_WORK_PASSED/posefusion/reg/keli_full_wm/keli_iphone_poseN/images，逐帧前景像素和全部相机矩阵完全一致。

背景来源、裁剪和相机核对记录见根目录 provenance；逐帧哈希见 case_manifest.json。整理结束前 E 盘中间恢复目录已不在原路径，已保存副本与此前记录的哈希全部一致。

# Guardian (guardian)

真实拍摄数据，三个 pose 的 image 均为带实际背景的 RGB PNG。配套 mask 与 NeuS cameras_sphere.npz 保持原帧序。

| Pose | 图像 / mask 数 | 格式 |
|---|---:|---|
| 1 | 60 | 1024x1024 RGB |
| 2 | 60 | 1024x1024 RGB |
| 3 | 60 | 1024x1024 RGB |

没有独立测量的配准 RT / mesh GT；未将预测结果当作 GT。

Original Gardian 60/60/60 frames retained. Background images recovered from raw_video/realworld_datasets_2/sliced_images_sliced; camera matrices and foreground pixels verified exactly equal.

背景来源、裁剪和相机核对记录见根目录 provenance；逐帧哈希见 case_manifest.json。整理结束前 E 盘中间恢复目录已不在原路径，已保存副本与此前记录的哈希全部一致。

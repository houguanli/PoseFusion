# honkai_vox

`honkai_vox` 是一个基于 Voxurf 的神经表面重建与跨场景刚体配准工程。当前仓库的重点不再是上游 Voxurf 的通用训练说明，而是：

- 训练并加载两个独立 case 的表面场
- 使用粗配准初始化 source/target 的相对位姿
- 基于相似相机对和 eLoFTR 匹配建立跨场景对应
- 沿相机射线从 SDF 场中提取成对零值面点
- 通过双边 SDF 约束进行精配准

## 核心流程

当前配准入口是 [mp_reg.py](/home/aoki/honkai_vox/mp_reg.py)。

整体流程如下：

1. 读取两个 case 的配置、数据目录、mesh 路径和训练好的 `surf_last.tar`
2. 加载 source / target 的神经场与数据集
3. 使用 coarse registration 结果初始化 `delta_RT`
4. 根据相机外参筛选相似视角图像对
5. 对 source 图像做 roll 补偿后，调用 eLoFTR 提取 2D 匹配
6. 将匹配点反投影为相机射线，并在 AABB 内沿射线查找 SDF 符号翻转
7. 得到 source / target 的成对零值面点集
8. 用双边 SDF + robust loss 优化 `source -> target` 的刚体变换
9. 导出 coarse / fine 结果的点云与 mesh debug 文件

## 关键文件

- [mp_reg.py](/home/aoki/honkai_vox/mp_reg.py)：多阶段配准主流程
- [LoFTR_helper.py](/home/aoki/honkai_vox/LoFTR_helper.py)：相机对筛选、roll 估计、eLoFTR 推理、匹配可视化
- [obb_reg_helper.py](/home/aoki/honkai_vox/obb_reg_helper.py)：粗配准辅助逻辑
- [run.py](/home/aoki/honkai_vox/run.py)：Voxurf 训练/渲染入口
- [configs/reg](/home/aoki/honkai_vox/configs/reg)：配准配置
- [verify_zero_sampling_bunny.py](/home/aoki/honkai_vox/verify_zero_sampling_bunny.py)：零值面提取验证脚本
- [ELOFTR_NOTES.md](/home/aoki/honkai_vox/ELOFTR_NOTES.md)：当前 eLoFTR 集成与兼容改动说明

## 当前精配准实现

当前保留的稳定版本为：

- 双边 SDF 约束
- `robust_loss_pytorch.AdaptiveLossFunction`
- `raw_translation` / `raw_quaternion` 学习率 `1e-4`
- `train_iters = 5000`

优化目标是：

- `source_zero_points --T--> target_space` 后查询 `target sdf`
- `target_zero_points --T^{-1}--> source_space` 后查询 `source sdf`
- 对两侧 `abs(sdf).mean()` 做 robust 包装
- 叠加四元数单位化正则

目前没有保留成对 3D 点一致性项，因为在当前数据上它没有带来更优的综合结果。

## eLoFTR 集成说明

本仓库不是直接照搬上游 eLoFTR，而是做了两层本地集成：

1. [LoFTR_helper.py](/home/aoki/honkai_vox/LoFTR_helper.py)
- 支持根据相机相对姿态估计 source 图像的 roll
- 在 eLoFTR 前旋转 source image / mask
- 匹配后将点坐标逆旋回原图坐标
- 增加旧版 Torch / Lightning 兼容补丁
- 反序列化 eLoFTR checkpoint 时注入 stub 模块，避免训练期依赖链污染推理

2. [dependencies/eloftr](/home/aoki/honkai_vox/dependencies/eloftr)
- 移除了推理路径对 `pytorch_lightning` profiler 的强依赖
- 为 `rank_zero_only` 增加 fallback
- 保证旧环境下只跑推理所需部分，不再因为 Lightning/Fabric 版本不兼容而在导入或 `torch.load()` 时崩溃

详细改动见 [ELOFTR_NOTES.md](/home/aoki/honkai_vox/ELOFTR_NOTES.md)。

## 配置

配准配置已经统一整理到 [configs/reg](/home/aoki/honkai_vox/configs/reg)：

- `configs/reg/dadian12.json`
- `configs/reg/caiwenji12.json`
- `configs/reg/fufu12.json`
- `configs/reg/fufu13.json`
- `configs/reg/monster12.json`
- `configs/reg/seal12.json`
- `configs/reg/tank12.json`
- `configs/reg/virtual.json`
- `configs/reg/example.json`

一个典型配置会指定：

- source / target 的实验目录
- source / target 的数据目录
- source / target 的 mesh 路径
- coarse registration 初值
- 数据分辨率、筛选阈值等

## 运行方式

### 1. 训练或渲染单个 case

沿用 Voxurf 主流程：

```bash
python run.py --config <config_folder>/fine.py -p <workdir> --scene <scene>
```

例如导出 mesh：

```bash
python run.py --config configs/custom_e2e/fine.py -p logs/custom/exp/scan --scene dadian_pose1 --render_only --mesh_from_sdf
```

### 2. 运行配准

```bash
python mp_reg.py --config ./configs/reg/dadian12.json
```

### 3. 验证零值面提取

```bash
python verify_zero_sampling_bunny.py
```

## Debug 输出

默认会在 [logs/debug/zero_points](/home/aoki/honkai_vox/logs/debug/zero_points) 写出配准过程的中间结果，包括：

- 零值面点云
- 匹配射线点云
- coarse RT 作用后的 source 点云
- fine RT 作用后的 source mesh
- target reference mesh
- 粗配准记录和 mesh debug record

常用文件包括：

- `*_pose1.ply` / `*_pose2.ply`
- `*_mesh_raw.ply`
- `*_mesh_coarse_rt.ply`
- `*_mesh_fine_rt.ply`
- `coarse_registration_record.json`
- `mesh_debug_record.json`

## 环境说明

当前本地验证环境依赖：

- Python 3.9
- CUDA
- 旧版 Torch 环境（`voxsurf`）
- `robust_loss_pytorch`
- OpenCV
- Open3D / trimesh

由于当前环境中的 `torch`、`pytorch_lightning`、`lightning_fabric` 不完全兼容，eLoFTR 推理路径已经做了本地补丁。直接替换环境版本前，请先阅读 [ELOFTR_NOTES.md](/home/aoki/honkai_vox/ELOFTR_NOTES.md)。

## 当前已知结论

- 零值面提取的关键问题不是 SDF 场本身异常，而是盒外采样被 `grid_sample` 默认零填充误判为零值面
- 当前零值面采样已经改为：
  - 只在 AABB 内采样
  - 只接受沿 ray 的真实符号翻转
  - 通过 bracket + refine 得到零值点
- 当前较稳定的精配准版本是双边 robust SDF，已经能在 `dadian12` 上收敛到较小的平移误差和很小的光轴误差

## 历史来源

本仓库以 Voxurf 为基础继续演化，保留了其训练、渲染和体素 SDF 表达部分。但当前主线工作已经围绕跨 case 配准、eLoFTR 集成和零值面采样展开，不再适合沿用上游 README。

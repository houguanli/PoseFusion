# eLoFTR 改动说明

这份说明专门记录当前仓库为了让 eLoFTR 在 `voxsurf` 运行环境中稳定服务于配准流程而做的改动。

## 改动目标

当前工程对 eLoFTR 的要求不是训练，而是：

- 在旧版 Torch 环境下稳定完成推理
- 支持 source 图像的 roll 补偿
- 输出可直接用于零值面提取的成对匹配点
- 尽量避免引入 `pytorch_lightning` / `lightning_fabric` / `torch._dynamo` 的训练期依赖

## 改动分布

### 1. 上层集成：[LoFTR_helper.py](LoFTR_helper.py)

这一层负责把 eLoFTR 接进配准主流程。

主要改动：

- 根据相机姿态估计 `pose_estimated_roll_deg`
- 将 `source_roll_deg` 应用到 source 图像与 mask
- 匹配结束后将 keypoints 旋回原图坐标
- 支持写出 `tmp/roll_debug.json`
- 在推理入口安装兼容补丁，避免旧版 Torch / Lightning 依赖链炸掉

具体兼容函数：

- `_patch_torch_lr_scheduler_compat()`
  - 为旧版 `torch.optim.lr_scheduler` 补 `LRScheduler` 别名
- `_install_lightning_stub_modules()`
  - 在 `torch.load()` 反序列化 checkpoint 之前注入 stub 模块
  - 规避 `pytorch_lightning -> lightning_fabric -> torch._dynamo` 的导入链

### 2. 下层兼容：[dependencies/eloftr](dependencies/eloftr)

这一层修改的是 eLoFTR 子仓库内部的推理依赖。

#### `src/utils/misc.py`

问题：

- 原始实现直接 `from pytorch_lightning.utilities import rank_zero_only`
- 在当前环境中，这会拖入不兼容的 Lightning 依赖

处理：

- 用 `try/except` 包装导入
- 导入失败时提供一个 no-op 的 `rank_zero_only`

目的：

- 对推理路径来说，`rank_zero_only` 只是一个轻量装饰器
- 不值得为了它加载整条 Lightning 训练栈

#### `src/utils/profiler.py`

问题：

- 原始实现依赖 `pytorch_lightning.profiler`
- 这会在推理时把工程重新绑回 Lightning

处理：

- 用本地实现替换 `PassThroughProfiler` / `SimpleProfiler`
- 保留 `InferenceProfiler`
- 提供一个最小 `PyTorchProfiler` stub
- `build_profiler()` 继续保留原接口

目的：

- 让 eLoFTR 在推理阶段只使用实际需要的 profiler 功能
- 不再对 Lightning 训练框架有硬依赖

## 为什么不能只改一处

只改 `LoFTR_helper.py` 不够，原因有两类：

1. 导入期问题
- `dependencies/eloftr` 内部文件会主动 import Lightning 工具
- 所以即使外层脚本没显式 import，仍然会在模块加载时崩溃

2. checkpoint 反序列化问题
- `eloftr_outdoor.ckpt` 是训练阶段保存的 checkpoint
- `torch.load()` 反序列化时可能访问训练期类路径
- 如果不提前安装 stub 模块，仍然会把 Lightning/Fabric 拉起来

因此当前方案是：

- 外层 helper 负责导入链和 checkpoint 兼容
- 内层 eLoFTR 代码负责去除推理期不必要的 Lightning 依赖

## 与配准逻辑的关系

这部分改动是配准链条里的关键前提。

因为当前 `mp_reg.py` 的精配准依赖：

1. `filter_and_collect_pose_pairs()` 选出相似视角
2. `run_loftr_inference(..., use_eloftr=True)` 生成跨 case 匹配
3. 用匹配点建立成对射线
4. 沿射线提取零值面点
5. 基于这些点做 fine registration

如果 eLoFTR 不能在本地环境里稳定跑起来，后续零值面和精配准都无法验证。

## 当前边界

这些改动只保证：

- eLoFTR 推理在当前 `voxsurf` 环境下可用
- `mp_reg.py` 能顺利跑通匹配、零值面提取与 fine registration

它们不保证：

- 上游 `dependencies/eloftr/train.py`
- 上游 `dependencies/eloftr/test.py`
- Lightning 训练工作流

也就是说，当前是“保推理，不保训练”。

## 后续建议

如果以后需要把这部分分享给其他机器或整理到远端仓库，建议按下面顺序收敛：

1. 给 `dependencies/eloftr` 建立自己的可控远端或 fork
2. 把当前兼容补丁整理成独立提交
3. 在父仓库记录对应的子仓库提交号
4. 再决定是否继续保留 stub 方案，或升级到兼容的 Torch/Lightning 版本

# PoseFusion — Voxurf base

本项目提供多姿态物体重建、刚体配准和多视图数据融合流程。输入位于同级 `../dataset`，数据版本为 `background_inputs_20260929`，共 **18 个 case、54 个 pose、2139 张图像及对应 mask**。其中虚拟数据为 11 个 case、1350 张图像，真实数据为 7 个 case、789 张图像。

## 目录结构

```text
2public/
├── dataset/
│   └── <case_id>/
│       ├── pose1/                  # pose2、pose3 结构相同
│       │   ├── image/000.png       # 带背景 RGB
│       │   ├── mask/000.png        # 物体轮廓
│       │   └── cameras_sphere.npz  # NeuS/IDR 相机
│       └── gt/                    # 仅虚拟 case 提供刚体 GT
└── code/
    ├── pipeline.py                # 统一运行入口
    ├── run.py                     # Voxurf 重建及 mesh 导出
    ├── mp_reg.py                  # OBB 初始化、特征匹配与 SDF 配准
    ├── fuse_registrations.py       # 配准后的图像与相机融合
    ├── evaluate_registration.py   # 虚拟 case 的 R/t 误差评估
    ├── project_paths.py           # 相对路径解析
    ├── generate_configs.py        # 根据 dataset manifest 生成配置
    ├── validate_release.py        # 数据和配置检查
    ├── configs/
    │   ├── dataset.json           # case 名称、帧数、尺寸、路径索引
    │   ├── cases/<case>/poseN/     # coarse.py、fine.py，共 108 份
    │   └── reg/<case>/            # 配准模板，共 54 份 JSON
    ├── lib/                       # 读取器、模型及 CUDA 扩展源码
    └── dependencies/              # eLoFTR 源码及匹配网络权重
```

`outputs/` 在运行时按需生成，不包含在初始输入中。数据集的来源、格式和 GT 说明见 [dataset/README.md](../dataset/README.md)。上游项目说明见 [UPSTREAM_README.md](UPSTREAM_README.md)，eLoFTR 集成说明见 [ELOFTR_NOTES.md](ELOFTR_NOTES.md)。

## 环境安装

### 运行条件

使用 Linux 或 WSL2 Ubuntu；版本基准为 Ubuntu 22.04、Python 3.9.21、PyTorch 2.0.1+cu117、torchvision 0.15.2、CUDA Toolkit 11.7、GCC/G++ 10.4。重建与配准需要 NVIDIA CUDA GPU；数据读取和 fusion 可在 CPU 上运行。CUDA Toolkit 必须包含 `nvcc`，仅安装 PyTorch 的 CUDA 运行库不足以编译 `lib/cuda` 扩展。

以下命令在 Linux/WSL 的 Bash 终端中执行。先进入 `code` 目录；本机路径为：

```bash
cd /mnt/f/HGL_WORK_PASSED/posefusion/2public/code
```

### 新建环境

先安装 Miniconda/Anaconda 并初始化 Bash，然后执行：

```bash
# Ubuntu 系统库及 Git；仅首次安装时需要
sudo apt-get update
sudo apt-get install -y git libgl1 libglib2.0-0

# Python、编译器和包含 nvcc 的 CUDA Toolkit
conda env create -f environment_voxsurf6.yml
conda activate posefusion

# 固定数值库及构建工具版本
python -m pip install 'numpy==1.26.0' 'setuptools==59.5.0' wheel

# PyTorch / torchvision，CUDA 11.7 版本
python -m pip install 'torch==2.0.1' 'torchvision==0.15.2' \
    --index-url https://download.pytorch.org/whl/cu117

# 与 PyTorch 2.0 / CUDA 11.7 匹配的 torch_scatter 二进制包
python -m pip install 'torch-scatter==2.1.1' --only-binary=:all: \
    --no-index --find-links 'https://data.pyg.org/whl/torch-2.0.0+cu117.html'

# 其余依赖，包括固定提交版本的 robust_loss_pytorch
python -m pip install -r requirements.txt
```

安装顺序为 PyTorch → torch_scatter → 其余依赖。`requirements.txt` 固定当前运行所用版本；`robust_loss_pytorch` 通过 Git 安装，因此需要 Git 和网络。eLoFTR 仅用于推理，无需另外执行其训练环境的安装脚本。

安装来源：[PyTorch 历史版本](https://pytorch.org/get-started/previous-versions/)、[PyG 扩展 wheel](https://data.pyg.org/whl/torch-2.0.0%2Bcu117.html)、[CUDA Toolkit 开发包](https://anaconda.org/conda-forge/cudatoolkit-dev)、[robust_loss_pytorch](https://github.com/jonbarron/robust_loss_pytorch/tree/9831f1db8006105fe7a383312fba0e8bd975e7f6)。

### 使用已有环境

本机已有 `voxsurf6` 环境时，可直接激活，无需重新安装：

```bash
source /home/aoki/miniconda3/etc/profile.d/conda.sh
conda activate voxsurf6
```

### 终端环境与依赖检查

每次打开终端后激活环境，并设置：

```bash
# 使用当前环境中的 nvcc；不要覆盖为另一版本的系统 CUDA
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
if [ -d /usr/lib/wsl/lib ]; then
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
fi

# 避免在代码目录生成 Python 字节码缓存
export PYTHONDONTWRITEBYTECODE=1

nvcc --version
nvidia-smi
python -B - <<'PY'
import torch, torchvision, mmengine, kornia, einops, loguru
import torch_scatter, torch_efficient_distloss, robust_loss_pytorch
print('torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('torchvision:', torchvision.__version__, 'CUDA available:', torch.cuda.is_available())
print('mmengine:', mmengine.__version__, 'kornia:', kornia.__version__)
print('einops:', einops.__version__, 'loguru:', loguru.__version__)
assert torch.cuda.is_available(), '重建/配准需要 CUDA GPU'
PY
```

首次运行重建或配准时，会编译 CUDA 扩展，默认缓存位于用户的 `~/.cache/torch_extensions`。Kornia 0.4.1、einops 0.3.0 和 loguru 0.5.3 对应当前集成版本，不建议单独升级。

本目录已包含以下推理权重：

| 网络 | 文件 |
|---|---|
| eLoFTR | `dependencies/eloftr/weights/eloftr_outdoor.ckpt` |
| ResNet18 | `dependencies/resnet18-f37072fd.pth` |

## 数据和配置检查

```bash
# 列出指定 case 的配置、输入路径和输出路径
python -B pipeline.py plan --case seal

# 检查输入目录及已有重建文件；不执行训练
python -B pipeline.py preflight --case seal

# 检查全部数据、相机、mask、配置及文件哈希
python -B pipeline.py validate --checksums

# 只打印即将执行的命令
python -B pipeline.py train --case seal --pose 1 --dry-run
python -B pipeline.py register --case seal --dry-run
python -B pipeline.py fuse --case seal --dry-run
```

`validate` 的报告写入 `outputs/validation/`，其临时检查文件位于 `outputs/validation_tmp/`。`plan`、`preflight` 和 `--dry-run` 不运行训练。尚未重建时，`preflight` 列出的 checkpoint/mesh 缺失是正常状态；准备配准时应使用 `--require-results` 严格检查。

所有配置相对路径均以 `code` 为基准，默认读取同级 `../dataset`。移动完整的 `2public` 目录后不必修改配置。若数据在其他位置，可设置相同版本数据集的绝对路径：

```bash
export POSEFUSION_DATASET=/absolute/path/to/dataset
```

数据集已有完整配置，常规运行不需要执行 `generate_configs.py`。修改数据目录结构或 manifest 后，才需重新生成配置；该脚本会覆盖 `configs/dataset.json`、各 pose 的重建配置及配准模板。

## 运行步骤

以下以 `seal` 为例。`--case all` 选择全部 18 个 case，也可传入多个 ID，例如 `--case bunny book`。

### 1. 单 pose 重建与 mesh 导出

```bash
# 依次重建一个 case 的三个 pose
python -B pipeline.py train --case seal

# 或只重建 pose1
python -B pipeline.py train --case seal --pose 1

# 或分阶段执行：先 coarse，再 fine
python -B pipeline.py train --case seal --pose 1 --stage coarse
python -B pipeline.py train --case seal --pose 1 --stage fine

# 已有 fine checkpoint 时，单独导出 mesh
python -B pipeline.py mesh --case seal
```

统一入口在 coarse 阶段自动执行 DVGO 初始化。默认迭代次数为 DVGO 10000、Voxurf coarse 10000、Voxurf fine 20000。fine 需要同一 pose 的 `coarse/coarse_last.tar` 和 `coarse/surf_last.tar`；不能从空目录直接进入 fine。coarse 使用全部图像，fine 保留固定验证索引。

输出位置：

```text
outputs/reconstruction/<case>/poseN/
├── coarse/
│   ├── coarse_last.tar
│   └── surf_last.tar
└── fine/
    ├── surf_last.tar
    └── meshes/
        ├── surface_normalized.ply  # 配准输入，与 SDF 坐标一致
        └── surface_world.ply       # 世界坐标下的 mesh
```

重建完成后会导出最终 mesh；`mesh` 命令可补导出。使用 per-pose 配置时无需额外指定 `--scene` 或 `--prefix`。已有 checkpoint 时沿用 `run.py` 的恢复行为；需要从头运行时应选择独立输出位置，不混用不同数据版本的 checkpoint。

### 2. 刚体配准

```bash
# 确认三个 pose 的 fine checkpoint 和 normalized mesh 都存在
python -B pipeline.py preflight --case seal --require-results

# 执行三组配准
python -B pipeline.py register --case seal
```

每个 case 的三组关系为：

| 配准模板 | source → target | 用途 |
|---|---|---|
| `pose1_to_pose2.json` | pose1 → pose2 | 融合到 pose2 |
| `pose3_to_pose2.json` | pose3 → pose2 | 融合到 pose2 |
| `pose1_to_pose3.json` | pose1 → pose3 | 第三组 pose 组合 |

变换约定为列向量 **`x_target = T @ x_source`**。配准使用新重建的 SDF 和 `surface_normalized.ply`，通过 OBB 产生粗初始化，再执行特征匹配及 SDF 精配准。模板不含 GT 初始化或旧预测变换。

入口将模板复制到 `outputs/registration/<case>/`，估计的 `raw_reg`、`fine_rt`、耗时等写入运行副本；`configs/reg/` 模板保持不变。调试结果位于 `outputs/registration_debug/` 和 `outputs/matching_debug/`。再次运行会沿用运行副本中的粗配准缓存；若需重新估计，可在运行副本设置 `force_recompute_raw_reg=true`。直接调用 `mp_reg.py --config ...` 会改写指定 JSON，应传入运行副本。

### 3. Fusion：图像与相机融合

```bash
# 使用 pose1→pose2、pose3→pose2 的配准结果
python -B pipeline.py fuse --case seal
```

输出为 `outputs/fusion/seal_fused_pose2/`：

```text
image/                 # 目标视图及保留的源视图，重新连续编号
mask/
cameras_sphere.npz     # 变换到 pose2 坐标的 NeuS 相机
fusion_summary.json    # 帧数、过滤统计、配置来源
fusion_metadata.json   # 新帧号与原始 pose/帧号的对应关系
```

目标 pose2 的图像全部保留。源视图与目标相机的角度及归一化距离共同计算重叠程度；默认阈值为 `0.5`，过滤已有相近视角，保留补充视图。默认 fusion 优先使用 `fine_rt`，缺少它时使用 `raw_reg`；两者都缺少则报错。正常流程应先完成精配准，再执行 fusion。

已有同名输出会报错，不会自动删除。调整阈值或另存输出时可直接调用：

```bash
python -B fuse_registrations.py \
    --configs outputs/registration/seal/pose1_to_pose2.json \
              outputs/registration/seal/pose3_to_pose2.json \
    --output_name seal_fused_pose2_alt \
    --similarity_threshold 0.5
```

该命令产生融合数据集，不会自动重建最终 mesh。融合相机采用单位 `scale_mat`；源投影变换为 `P_source_normalized @ inverse(T)`，RGB 和 mask 与重编号后的相机一一对应。

### 4. 融合数据的最终重建（可选）

如需最终融合 mesh，先完成上一节的 fusion，再为融合帧数生成单独配置。以下命令只创建运行配置：

```bash
python -B - <<'PY'
import json
from project_paths import ROOT

case = 'seal'
fused = ROOT / 'outputs/fusion' / f'{case}_fused_pose2'
count = json.loads((fused / 'fusion_summary.json').read_text())['total_count']
config_dir = ROOT / 'outputs/fused_configs' / case
config_dir.mkdir(parents=True, exist_ok=True)
for stage in ['coarse', 'fine']:
    base = ROOT / 'configs/cases' / case / 'pose2' / f'{stage}.py'
    text = (f'_base_ = {str(base)!r}\n'
            f'expname = {case + "/fused_pose2"!r}\n'
            f'expected_image_count = {count}\n'
            f'data = dict(datadir={str(fused)!r})\n')
    (config_dir / f'{stage}.py').write_text(text, encoding='utf-8')
PY

python -B run.py --config outputs/fused_configs/seal/coarse.py \
    --run_dvgo_init --sdf_mode voxurf_coarse
python -B run.py --config outputs/fused_configs/seal/fine.py \
    --sdf_mode voxurf_fine
```

输出位于 `outputs/reconstruction/seal/fused_pose2/`，不会覆盖三个输入 pose 的重建结果。其他 case 将脚本中的 `case` 和命令中的目录名称对应修改即可。

### 5. 虚拟数据的 R/t 误差评估

```bash
python -B pipeline.py evaluate --case seal
```

评估读取当前 `dataset/<case>/gt/transforms.json` 中的 `relative_transforms`，先将预测从归一化坐标转换回世界坐标，再输出旋转角误差和世界单位的平移 L2 误差。GT 仅用于评估。seal 使用已校正 pose1 原点的当前版本 GT。真实 case 没有相应刚体 GT 时跳过评估；数据读写和 fusion 格式检查不等同于配准精度验证。

## 相机与 mask 约定

相机采用 **NeuS/IDR format**：`p ~ world_mat_i @ scale_mat_i @ X`。分解得到 OpenCV c2w：**+Z 向前、+Y 向下**，配置为 `inverse_y=True`。`world_mat_inv_i` 等为辅助矩阵，不作为额外帧。读取器保留逐帧内参，缩放图像时同步缩放 fx、fy、cx、cy；mask 使用最近邻插值。

主图像为带背景 RGB。重建配置的 `data.mode.wmask=True` 会在内存中将 mask 外的像素设为黑色；配准读取原始 RGB 并用 mask 排除背景特征。原始数据文件不改写。mask 表示物体轮廓，不包含地面、阴影和倒影；虚拟数据保留物体轮廓语义，不作为完整场景遮挡分割使用。

配准使用归一化 mesh，与 SDF 一致；`surface_world.ply` 用于世界坐标比较。当前数据的 54 组 `scale_mat` 均为单位矩阵，两种坐标一致，代码仍保留坐标转换。

## Case 列表

| case ID | 类型 | pose1 / pose2 / pose3 图像数 | 宽×高 |
|---|---|---|---|
| bunny | 虚拟 | 40 / 40 / 40 | 800×800 |
| book | 虚拟 | 45 / 45 / 45 | 800×800 |
| nikon | 虚拟 | 40 / 40 / 40 | 800×800 |
| heel_slipper | 虚拟 | 40 / 40 / 40 | 800×800 |
| figurine_a | 虚拟 | 45 / 45 / 45 | 800×800 |
| tank | 虚拟 | 40 / 40 / 40 | 800×800 |
| monster | 虚拟 | 40 / 40 / 40 | 1200×800 |
| figurine_b | 虚拟 | 40 / 40 / 40 | 800×800 |
| rubik | 虚拟 | 40 / 40 / 40 | 800×800 |
| lego | 虚拟 | 40 / 40 / 40 | 800×800 |
| seal | 虚拟 | 40 / 40 / 40 | 800×800 |
| figurine_c | 真实 | 32 / 32 / 32 | 1024×1024 |
| figurine_d | 真实 | 33 / 32 / 33 | 1024×1024 |
| figurine_e | 真实 | 32 / 33 / 32 | 1024×1024 |
| figurine_f | 真实 | 55 / 36 / 35 | 1920×1080 |
| joystick | 真实 | 32 / 32 / 32 | 1024×1024 |
| airplane | 真实 | 32 / 32 / 32 | 1024×1024 |
| guardian | 真实 | 60 / 60 / 60 | 1024×1024 |

`figurine_f` 对应 Keli，`guardian` 对应历史 `Gardian` 拼写，`heel_slipper` 对应 shoe，`rubik` 对应 rubic。seal 每个 pose 为 40 张；monster 和 Keli 使用各自的非方形尺寸。完整映射以 `configs/dataset.json` 为准。

## 常见问题

- **找不到数据或 case**：检查 `code` 与 `dataset` 是否同级，或 `POSEFUSION_DATASET` 是否设置正确；case ID 使用上表名称。
- **缺少 fine checkpoint/mesh**：先完成 coarse/fine 重建，必要时运行 `pipeline.py mesh`。配准输入是预测 mesh，不是 `dataset/gt/meshes`。
- **提示没有 raw_reg/fine_rt**：fusion 需要配准结果，应读取 `outputs/registration/` 的运行副本，而非空白模板。
- **CUDA 编译失败**：检查 `nvcc`、`CUDA_HOME`、GCC/G++ 和 PyTorch CUDA 版本是否匹配；不要将当前环境的 CUDA 11.7 覆盖为其他系统版本。
- **WSL 找不到 libcuda.so**：检查 `/usr/lib/wsl/lib` 是否存在且已加入 `LD_LIBRARY_PATH`。
- **GPU 显存不足**：可在对应 pose 的配置中提高 `reso_level` 或降低 `surf_train.N_rand`；这些修改会改变运行设置，应在实验记录中保留。
- **fusion 输出已存在**：用不同的 `--output_name` 保存新结果，不自动覆盖已有融合数据。

各 case 的 OBB 和 drop-worst-face 设置见配准 JSON。不同重建设置和数据版本的结果不能直接混用；论文数值是否复现，需要另行完成对应实验与评估。

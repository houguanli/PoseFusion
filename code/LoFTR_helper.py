"""
this file implement the interface for LoFTR and eLoFTR, which can detect key points pair for reg

"""

import json
import math
import os
import pickle
import sys
import types
from pathlib import Path
from project_paths import ROOT

import cv2
import cv2 as cv
import matplotlib.cm as cm
import numpy as np
import open3d as o3d
import torch

from dependencies.eloftr.src.utils.plotting import make_matching_figure


def _patch_torch_lr_scheduler_compat():
    lr_scheduler = torch.optim.lr_scheduler
    if not hasattr(lr_scheduler, "LRScheduler") and hasattr(lr_scheduler, "_LRScheduler"):
        lr_scheduler.LRScheduler = lr_scheduler._LRScheduler


class _LightningStub:
    def __init__(self, *args, **kwargs):
        pass


def _install_lightning_stub_modules():
    def ensure_module(name):
        if name in sys.modules:
            return sys.modules[name]
        module = types.ModuleType(name)

        def _module_getattr(attr):
            return _LightningStub

        module.__getattr__ = _module_getattr
        sys.modules[name] = module
        return module

    module_names = [
        "pytorch_lightning",
        "pytorch_lightning.callbacks",
        "pytorch_lightning.callbacks.model_checkpoint",
        "pytorch_lightning.core",
        "pytorch_lightning.core.module",
        "pytorch_lightning.utilities",
        "pytorch_lightning.utilities.types",
        "lightning_fabric",
        "lightning_fabric.wrappers",
        "lightning_fabric.utilities",
        "lightning_fabric.utilities.types",
        "torch._dynamo",
    ]
    for module_name in module_names:
        ensure_module(module_name)

    sys.modules["pytorch_lightning"].callbacks = sys.modules["pytorch_lightning.callbacks"]
    sys.modules["pytorch_lightning.callbacks"].model_checkpoint = sys.modules["pytorch_lightning.callbacks.model_checkpoint"]
    sys.modules["pytorch_lightning"].core = sys.modules["pytorch_lightning.core"]
    sys.modules["pytorch_lightning.core"].module = sys.modules["pytorch_lightning.core.module"]
    sys.modules["pytorch_lightning"].utilities = sys.modules["pytorch_lightning.utilities"]
    sys.modules["pytorch_lightning.utilities"].types = sys.modules["pytorch_lightning.utilities.types"]
    sys.modules["lightning_fabric"].wrappers = sys.modules["lightning_fabric.wrappers"]
    sys.modules["lightning_fabric"].utilities = sys.modules["lightning_fabric.utilities"]
    sys.modules["lightning_fabric.utilities"].types = sys.modules["lightning_fabric.utilities.types"]

    sys.modules["pytorch_lightning.callbacks.model_checkpoint"].ModelCheckpoint = _LightningStub
    sys.modules["pytorch_lightning.core.module"].LightningModule = _LightningStub
    sys.modules["lightning_fabric.wrappers"]._unwrap_objects = _LightningStub
    sys.modules["torch._dynamo"].OptimizedModule = _LightningStub


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("pytorch_lightning") or module.startswith("lightning_fabric"):
            return _LightningStub
        if module == "torch._dynamo":
            return _LightningStub
        return super().find_class(module, name)


class _CompatPickleModule:
    Unpickler = _CompatUnpickler
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)
    dump = staticmethod(pickle.dump)
    dumps = staticmethod(pickle.dumps)

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def rotate_image_keep_size(image: np.ndarray, angle_deg: float, interpolation, border_value=0):
    h, w = image.shape[:2]
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        rot_mat,
        (w, h),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )
    return rotated


def rotate_points_about_image_center(points: np.ndarray, angle_deg: float, image_size):
    if points.size == 0:
        return points
    w, h = image_size
    center = np.array([(w - 1) / 2.0, (h - 1) / 2.0], dtype=np.float32)
    theta = np.deg2rad(angle_deg)
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ], dtype=np.float32)
    return (points - center) @ rot.T + center


def normalize_angle_deg(angle_deg: float) -> float:
    angle_deg = (angle_deg + 180.0) % 360.0 - 180.0
    return float(angle_deg)


def write_roll_debug_json(results, delta_RT, npz1_path, npz2_path, debug_path="tmp/roll_debug.json", verbose=False):
    debug_file = Path(debug_path)
    debug_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "npz1_path": npz1_path,
        "npz2_path": npz2_path,
        "delta_RT": np.asarray(delta_RT, dtype=np.float32).tolist(),
        "pairs_total": len(results),
        "pairs": []
    }
    for pair in results:
        payload["pairs"].append({
            "idx1": int(pair["idx1"]),
            "idx2": int(pair["idx2"]),
            "img1": pair["img1"],
            "img2": pair["img2"],
            "mask1": pair["mask1"],
            "mask2": pair["mask"],
            "pose_estimated_roll_deg": float(pair.get("pose_estimated_roll_deg", pair.get("source_roll_deg", 0.0))),
            "applied_source_roll_deg": float(pair.get("applied_source_roll_deg", pair.get("source_roll_deg", 0.0))),
            "roll_strategy": pair.get("roll_strategy", "negate_pose_estimate")
        })
    with open(debug_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"[roll-debug] wrote {debug_file} with {len(results)} pairs")

def evaluate_extrinsic_difference(
    M1: np.ndarray,
    M2: np.ndarray,
    rotation_threshold_deg: float = 30.0,
    translation_threshold: float = 0.5,
    is_print_info = False
) -> bool:
    """
    比较两个相机外参矩阵的差异，并判断是否在给定阈值范围内。

    参数:
    -------
    M1 : np.ndarray
        第一个外参矩阵，形状为 (4, 4)。
    M2 : np.ndarray
        第二个外参矩阵，形状为 (4, 4)。
    rotation_threshold_deg : float
        旋转角度阈值 (单位: 度)，用于判断两个矩阵的旋转差异是否可接受。
    translation_threshold : float
        平移距离阈值，用于判断两个矩阵的平移差异是否可接受。
    bool
        如果两个外参矩阵的差异（旋转与平移）都在阈值范围内，返回 True；否则返回 False。
    """

    # 1. 计算相对变换矩阵 (M_rel = M1^-1 * M2)
    #    其含义为：从 M1 对应的坐标系，变换到 M2 对应的坐标系
    M_rel = np.linalg.inv(M1) @ M2

    # 2. 提取旋转矩阵和平移向量
    R_rel = M_rel[:3, :3]   # 相对旋转
    t_rel = M_rel[:3, 3]    # 相对平移

    # 3. 计算旋转差异（以角度为单位）
    #    旋转矩阵 R 的旋转角度可通过以下公式计算：
    #    angle = arccos((trace(R) - 1) / 2)
    #    注意数值稳定性，可能需要 min/max 限制
    trace_val = np.trace(R_rel)
    # 防止浮点误差导致 arccos 参数超出 [-1, 1]
    cos_angle = max(min((trace_val - 1.0) / 2.0, 1.0), -1.0)
    rotation_angle_deg = np.degrees(np.arccos(cos_angle))

    # 4. 计算平移差异（向量范数）
    translation_diff = np.linalg.norm(t_rel)

    # 5. 根据阈值进行判断
    rotation_ok = (rotation_angle_deg <= rotation_threshold_deg)
    translation_ok = (translation_diff <= translation_threshold)
    if rotation_ok and translation_ok and is_print_info:
        print("rotation diff: ", rotation_angle_deg)
        print("translation diff ", translation_diff)
    return rotation_ok and translation_ok

def test_similar_pose_detect():
    M1_example = [[0.7423551082611084, -0.2518492341041565, -0.6208710670471191, 0.49617502093315125],
                  [-0.6630696654319763, -0.14315412938594818, -0.7347418069839478, 0.22048130631446838],
                  [0.09616389870643616, 0.9571200609207153, -0.2732647955417633, 0.20496077835559845],
                  [0.0, 0.0, 0.0, 1.0]]

    M2_example = [[-0.9261959195137024, 0.004189362749457359, -0.37701937556266785, 0.2073606550693512],
                  [0.37704265117645264, 0.010291064158082008, -0.926138699054718, 0.5093762874603271],
                  [-2.3283061589829401e-10, -0.9999382495880127, -0.011111109517514706, 0.0061111110262572765],
                  [0.0, 0.0, 0.0, 1.0]]

    RT = np.array([
        [
            -0.5849950772807017,
            0.801868264500357,
            -0.12160610981579512,
            0.2416590098834332
        ],
        [
            0.8033941419548528,
            0.5934671410150251,
            0.04852427442093907,
            -0.02324493170884405
        ],
        [
            0.11107930603839655,
            -0.06931117458706122,
            -0.9913916223407326,
            0.023850916562664706
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0
        ]
    ])  # a test for obb based exc finding
    M1_example = RT @ M1_example  # trans with raw RT
    result = evaluate_extrinsic_difference(
        M1_example,
        M2_example,
        rotation_threshold_deg=30.0,
        translation_threshold=0.5
    )
    print("is differ:", result)

def from_npz_get_poses(npz_path):
    data = np.load(npz_path)
    poses = []
    keys = sorted((k for k in data.files if k.startswith('world_mat_') and not k.startswith('world_mat_inv_')),
                  key=lambda k: int(k.rsplit('_', 1)[1]))
    for key in keys:
        idx = int(key.rsplit('_', 1)[1])
        projection = data[key] @ data[f'scale_mat_{idx}']
        intrinsics, pose = load_K_Rt_from_P(None, projection[:3, :4])
        poses.append(pose)
    return poses

def find_near_camera_poses(
    npz_path1: str,
    npz_path2: str,
    delta_RT: np.array,
    rotation_threshold_deg: float = 50.0,
    translation_threshold: float = 2
):
    """
    从两个 npz 文件中读取所有相机位姿 (外参矩阵)，并找出在旋转和平移上都足够接近的配对。

    参数:
    -------
    npz_path1 : str
        第一个 .npz 文件路径。
    npz_path2 : str
        第二个 .npz 文件路径。
    rotation_threshold_deg : float
        旋转角度阈值 (单位: 度)，用于判断两个位姿的旋转差异是否可接受。
    translation_threshold : float
        平移距离阈值，用于判断两个位姿的平移差异是否可接受。

    返回:
    -------
    near_pairs : list of (int, int)
        一个列表，每个元素是 (i, j)，表示第 i 个 pose1 与第 j 个 pose2 满足阈值要求。
    """
    poses1 = from_npz_get_poses(npz_path1)  # shape: (N1, 4, 4)
    poses2 = from_npz_get_poses(npz_path2)  # shape: (N2, 4, 4)

    near_pairs = []
    for i, M1 in enumerate(poses1):
        for j, M2 in enumerate(poses2):
            delta_M1 = delta_RT @ M1
            if evaluate_extrinsic_difference(
                delta_M1, M2,
                rotation_threshold_deg=rotation_threshold_deg,
                translation_threshold=translation_threshold
            ):
                near_pairs.append((i, j))

    return near_pairs

def measure_extrinsic_difference(M1: np.ndarray, M2: np.ndarray) -> float:
    """
    计算两个外参矩阵 M1, M2 的差异度。
    这里简单地将 '旋转角度(度)' + '平移差异' 作为一个标量距离。
    """
    # 相对变换矩阵
    M_rel = np.linalg.inv(M1) @ M2
    R_rel = M_rel[:3, :3]
    t_rel = M_rel[:3, 3]

    # 计算旋转角度
    trace_val = np.trace(R_rel)
    # 为避免浮点误差，需确保 arccos 参数在 [-1, 1] 范围内
    cos_angle = max(min((trace_val - 1.0) / 2.0, 1.0), -1.0)
    rotation_angle_deg = np.degrees(np.arccos(cos_angle))

    # 计算平移差异
    translation_diff = np.linalg.norm(t_rel)

    # 将旋转与平移差异合并为单一标量
    # 可根据需求对二者加权
    difference = rotation_angle_deg * math.pi / 180 + translation_diff
    return difference

def find_K_near_camera_poses(
    npz_path1: str,
    npz_path2: str,
    delta_RT:torch.Tensor,
    K: int = 5
):
    """
    在不使用阈值的情况下，寻找两个 .npz 文件中外参矩阵差异最小的 K 个配对。

    参数:
    -------
    npz_path1 : str
        第一个 .npz 文件路径。
    npz_path2 : str
        第二个 .npz 文件路径。
    K : int
        需要返回的最小差异配对数量。

    返回:
    -------
    top_K_pairs : list of (int, int, float)
        长度最多为 K 的列表，每个元素为 (i, j, diff)，
        表示第 i 个 pose1 与第 j 个 pose2 的差异度为 diff。
        该列表按 diff 从小到大排序。
    """
    poses1 = from_npz_get_poses(npz_path1)  # shape: (N1, 4, 4)
    poses2 = from_npz_get_poses(npz_path2)  # shape: (N2, 4, 4)

    # 存储所有配对及其差异度
    pairs_diff = []
    for i, M1 in enumerate(poses1):
        for j, M2 in enumerate(poses2):
            M1_delta = delta_RT @ M1
            diff = measure_extrinsic_difference(M1_delta, M2)
            pairs_diff.append((i, j, diff))

    # 根据差异度升序排序
    pairs_diff.sort(key=lambda x: x[2])
    # 返回差异度最小的 K 个配对
    # 若总配对数小于 K，则只返回全部配对
    top_k_pairs = [(i, j) for i, j, _ in pairs_diff[:K]]
    return top_k_pairs

def run_loftr_inference(
    img0_path: str,
    img1_path: str,
    mask0_path: str = None,
    mask1_path: str = None,
    out_size_img=(1024, 1024),
    out_size_mask=(128, 128),
    use_eloftr = True,
    write_out_flag=True
):
    # 1) Depending on 'use_eloftr', import the corresponding LoFTR and default config.
    if use_eloftr:
        from dependencies.eloftr.src.loftr import LoFTR, full_default_cfg
        print("using eloftr")
        checkpoint_path: str = str(ROOT / "dependencies/eloftr/weights/eloftr_outdoor.ckpt")
        matcher = LoFTR(config=full_default_cfg)
    else:
        from dependencies.LoFTR.src.loftr import LoFTR, default_cfg
        checkpoint_path: str = "./dependencies/LoFTR/weights/indoor_ds_new.ckpt"
        matcher = LoFTR(config=default_cfg)
    state_dict = torch.load(checkpoint_path, weights_only=False)
    matcher.load_state_dict(state_dict['state_dict'], strict=True)
    matcher = matcher.eval().cuda()

    img0_raw = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    if img0_raw is None or img1_raw is None:
        print(img0_path, " ", img1_path)
        raise FileNotFoundError("无法读取图像，请检查路径是否正确。")

    h0_orig, w0_orig = img0_raw.shape[:2]
    h1_orig, w1_orig = img1_raw.shape[:2]

    # 3) 按指定大小缩放图像
    #    注意: cv2.resize 的参数顺序为 (width, height)
    img0_resized = cv2.resize(img0_raw, out_size_img)
    img1_resized = cv2.resize(img1_raw, out_size_img)

    # 4) 若有 mask，则读取并缩放
    use_mask = (mask0_path is not None) and (mask1_path is not None)
    if use_mask:
        mask0_raw = cv2.imread(mask0_path, cv2.IMREAD_GRAYSCALE)
        mask1_raw = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
        if mask0_raw is None or mask1_raw is None:
            raise FileNotFoundError("无法读取 mask，请检查路径是否正确。")

        mask0_resized = cv2.resize(mask0_raw, out_size_mask)
        mask1_resized = cv2.resize(mask1_raw, out_size_mask)
    else:
        mask0_resized, mask1_resized = None, None

    # 5) 转为 PyTorch 张量并归一化到 [0,1]
    #    shape: (1, 1, H, W)
    img0_t = torch.from_numpy(img0_resized)[None][None].float().cuda() / 255.
    img1_t = torch.from_numpy(img1_resized)[None][None].float().cuda() / 255.

    # 6) 组装 batch
    if use_mask:
        mask0_t = torch.from_numpy(mask0_resized)[None].cuda()
        mask1_t = torch.from_numpy(mask1_resized)[None].cuda()
        batch = {
            'image0': img0_t,
            'image1': img1_t,
            'mask0': mask0_t,
            'mask1': mask1_t
        }
    else:
        batch = {
            'image0': img0_t,
            'image1': img1_t
        }

    # 7) 推理，得到匹配结果
    with torch.no_grad():
        matcher(batch)

    # mkpts0_f, mkpts1_f, mconf 分别是匹配后的坐标与置信度
    mkpts0 = batch['mkpts0_f'].cpu().numpy()  # shape: (N, 2)
    mkpts1 = batch['mkpts1_f'].cpu().numpy()  # shape: (N, 2)
    mconf = batch['mconf'].cpu().numpy()      # shape: (N,)
    confidence_threshold = 0
    # 3) 根据阈值过滤
    valid_mask = (mconf >= confidence_threshold)

    mkpts0 = mkpts0[valid_mask]  # shape: (N_filtered, 2)
    mkpts1 = mkpts1[valid_mask]  # shape: (N_filtered, 2)
    mconf = mconf[valid_mask]  # shape: (N_filtered,)
    # print(len(mconf))

    # 8) mconf (out_size_img) 映射回原图分辨率
    #    out_size_img = (width_resize, height_resize)
    w_resize, h_resize = out_size_img
    scale_x0 = w0_orig / w_resize
    scale_y0 = h0_orig / h_resize
    mkpts0[:, 0] *= scale_x0
    mkpts0[:, 1] *= scale_y0

    scale_x1 = w1_orig / w_resize
    scale_y1 = h1_orig / h_resize
    mkpts1[:, 0] *= scale_x1
    mkpts1[:, 1] *= scale_y1

    # 9) 拼装输出坐标 => (N, 2, 2)
    #    matches[i, 0] = [x0, y0], matches[i, 1] = [x1, y1]
    matches = np.stack([mkpts0, mkpts1], axis=1).astype(np.uint)


    if write_out_flag:
        color = cm.jet(mconf, alpha=0.7)
        text = [
            'LoFTR',
            'Matches: {}'.format(len(mkpts0)),
        ]
        make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, mkpts0, mkpts1, text, path="./logs/debug/LoFTR/e-LoFTR-fufu_pose23_4paper.pdf")
    return matches, mconf


import os

def filter_and_collect_pose_pairs(
    npz1_path: str,
    images1_folder: str,
    masks1_folder: str,
    npz2_path: str,
    images2_folder: str,
    masks2_folder: str,
    delta_RT: torch.tensor,
    K: int = 0,
    file_ext: str = ".png"
):
    """
    Filters out all possible corresponding pose/image pairs between two .npz files
    (each containing multiple camera poses) and their corresponding image/mask folders.
    If K == 0, it only uses the threshold-based method (find_near_camera_poses).
    Otherwise, it uses both the threshold-based method and the top-K method
    (find_K_near_camera_poses) and merges the results.

    Args:
        npz1_path (str): Path to the first .npz file containing multiple camera poses.
        images1_folder (str): Path to the folder containing the images associated with npz1.
        masks1_folder (str): Path to the folder containing the masks associated with npz1.
        npz2_path (str): Path to the second .npz file containing multiple camera poses.
        images2_folder (str): Path to the folder containing the images associated with npz2.
        masks2_folder (str): Path to the folder containing the masks associated with npz2.
        delta_RT: Parameter(s) used by the extrinsic comparison logic
                  (e.g., rotation/translation thresholds). Actual type depends on your code.
        K (int, optional): If 0, only the threshold-based method is used. If > 0,
                           the threshold-based pairs are merged with the top-K pairs.
                           Defaults to 0.
        file_ext (str, optional): File extension for both images and masks. Defaults to ".png".

    Returns:
        list of dict: A list of dictionaries. Each dictionary describes a matching pair
                      of camera poses and their corresponding image/mask file paths. Example:
            {
                "idx1": i,  # index of pose i in npz1
                "idx2": j,  # index of pose j in npz2
                "img1": "<images1_folder>/<i>.png",
                "mask1": "<masks1_folder>/<i>.png",
                "img2": "<images2_folder>/<j>.png",
                "mask2": "<masks2_folder>/<j>.png"
            }
    """

    # 1) If K == 0, only use the threshold-based approach.
    #    Otherwise, use both threshold-based and top-K methods, then merge.
    if K == 0:
        # Only threshold-based pairs
        combined_pairs = find_near_camera_poses(npz1_path, npz2_path, delta_RT)
    else:
        # Top-K pairs
        top_k_pairs = find_K_near_camera_poses(npz1_path, npz2_path, delta_RT, K=K)
        # Merge results (remove duplicates using a set)
        combined_pairs = top_k_pairs
    results = []
    for (i, j) in combined_pairs: # assume pics are ranged as 000 version
        # Construct image and mask file names by index
        img1_path = os.path.join(images1_folder, f"{i:03}{file_ext}")
        mask1_path = os.path.join(masks1_folder, f"{i:03}{file_ext}")

        img2_path = os.path.join(images2_folder, f"{j:03}{file_ext}")
        mask2_path = os.path.join(masks2_folder, f"{j:03}{file_ext}")

        results.append({
            "idx1": i,
            "idx2": j,
            "img1": img1_path,
            "mask1": mask1_path,
            "img2": img2_path,
            "mask2": mask2_path
        })

    return results


def evaluate_extrinsic_difference(
    M1: np.ndarray,
    M2: np.ndarray,
    rotation_threshold_deg: float = 30.0,
    translation_threshold: float = 0.5,
    is_print_info=False
) -> bool:
    optical_axis1 = M1[:3, 2] / np.linalg.norm(M1[:3, 2])
    optical_axis2 = M2[:3, 2] / np.linalg.norm(M2[:3, 2])
    cos_angle = np.clip(np.dot(optical_axis1, optical_axis2), -1.0, 1.0)
    viewing_angle_deg = np.degrees(np.arccos(cos_angle))

    M_rel = np.linalg.inv(M1) @ M2
    translation_diff = np.linalg.norm(M_rel[:3, 3])

    rotation_ok = viewing_angle_deg <= rotation_threshold_deg
    translation_ok = translation_diff <= translation_threshold
    if rotation_ok and translation_ok and is_print_info:
        print("viewing direction diff: ", viewing_angle_deg)
        print("translation diff: ", translation_diff)
    return rotation_ok and translation_ok


def measure_extrinsic_difference(M1: np.ndarray, M2: np.ndarray) -> float:
    optical_axis1 = M1[:3, 2]
    optical_axis2 = M2[:3, 2]
    optical_axis1 = optical_axis1 / np.linalg.norm(optical_axis1)
    optical_axis2 = optical_axis2 / np.linalg.norm(optical_axis2)

    cos_angle = np.clip(np.dot(optical_axis1, optical_axis2), -1.0, 1.0)
    viewing_angle_deg = np.degrees(np.arccos(cos_angle))

    M_rel = np.linalg.inv(M1) @ M2
    translation_diff = np.linalg.norm(M_rel[:3, 3])
    return viewing_angle_deg + translation_diff


def estimate_relative_roll_deg(M1: np.ndarray, M2: np.ndarray) -> float:
    R1 = M1[:3, :3]
    R2 = M2[:3, :3]
    R_rel = R2.T @ R1
    roll_rad = np.arctan2(R_rel[1, 0], R_rel[0, 0])
    return float(np.degrees(roll_rad))


def find_K_near_camera_poses(
    npz_path1: str,
    npz_path2: str,
    delta_RT: torch.Tensor,
    K: int = 5
):
    if isinstance(delta_RT, torch.Tensor):
        delta_RT = delta_RT.detach().cpu().numpy()
    poses1 = from_npz_get_poses(npz_path1)
    poses2 = from_npz_get_poses(npz_path2)

    pairs_diff = []
    for i, M1 in enumerate(poses1):
        for j, M2 in enumerate(poses2):
            M1_delta = delta_RT @ M1
            diff = measure_extrinsic_difference(M1_delta, M2)
            pairs_diff.append((i, j, diff))

    pairs_diff.sort(key=lambda x: x[2])
    return [(i, j) for i, j, _ in pairs_diff[:K]]


def run_loftr_inference(
    img0_path: str,
    img1_path: str,
    mask0_path: str = None,
    mask1_path: str = None,
    out_size_img=(1024, 1024),
    out_size_mask=(128, 128),
    use_eloftr=True,
    write_out_flag=False,
    source_roll_deg: float = 0.0
):
    _patch_torch_lr_scheduler_compat()
    _install_lightning_stub_modules()
    if use_eloftr:
        from dependencies.eloftr.src.loftr import LoFTR, full_default_cfg
        if write_out_flag:
            print("using eloftr")
        checkpoint_path = str(ROOT / "dependencies/eloftr/weights/eloftr_outdoor.ckpt")
        matcher = LoFTR(config=full_default_cfg)
    else:
        from dependencies.LoFTR.src.loftr import LoFTR, default_cfg
        checkpoint_path = "./dependencies/LoFTR/weights/indoor_ds_new.ckpt"
        matcher = LoFTR(config=default_cfg)

    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location='cpu', pickle_module=_CompatPickleModule)
    except Exception:
        ckpt = torch.load(checkpoint_path, map_location='cpu', pickle_module=_CompatPickleModule)

    if 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt

    matcher.load_state_dict(state_dict, strict=True)
    matcher = matcher.eval().cuda()

    img0_raw = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img0_rgb = cv2.imread(img0_path, cv2.IMREAD_UNCHANGED)
    img1_rgb = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
    img0_rgb = cv2.cvtColor(img0_rgb, cv2.COLOR_BGR2RGB)
    img1_rgb = cv2.cvtColor(img1_rgb, cv2.COLOR_BGR2RGB)
    img0_rgb_unrotated = img0_rgb.copy()
    if img0_raw is None or img1_raw is None:
        print(img0_path, " ", img1_path)
        raise FileNotFoundError("无法读取图像，请检查路径是否正确。")

    h0_orig, w0_orig = img0_raw.shape[:2]
    h1_orig, w1_orig = img1_raw.shape[:2]

    if abs(source_roll_deg) > 1e-6:
        img0_raw = rotate_image_keep_size(img0_raw, source_roll_deg, cv2.INTER_LINEAR, border_value=0)
        img0_rgb = rotate_image_keep_size(img0_rgb, source_roll_deg, cv2.INTER_LINEAR, border_value=(0, 0, 0))

    img0_resized = cv2.resize(img0_raw, out_size_img)
    img1_resized = cv2.resize(img1_raw, out_size_img)

    use_mask = (mask0_path is not None) and (mask1_path is not None)
    if use_mask:
        mask0_raw = cv2.imread(mask0_path, cv2.IMREAD_GRAYSCALE)
        mask1_raw = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
        if mask0_raw is None or mask1_raw is None:
            raise FileNotFoundError("无法读取 mask，请检查路径是否正确。")
        if abs(source_roll_deg) > 1e-6:
            mask0_raw = rotate_image_keep_size(mask0_raw, source_roll_deg, cv2.INTER_NEAREST, border_value=0)
    else:
        mask0_raw, mask1_raw = None, None

    img0_t = torch.from_numpy(img0_resized)[None][None].float().cuda() / 255.
    img1_t = torch.from_numpy(img1_resized)[None][None].float().cuda() / 255.
    batch = {'image0': img0_t, 'image1': img1_t}

    with torch.no_grad():
        matcher(batch)

    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()

    valid_mask = mconf >= 0
    mkpts0 = mkpts0[valid_mask]
    mkpts1 = mkpts1[valid_mask]
    mconf = mconf[valid_mask]

    w_resize, h_resize = out_size_img
    mkpts0[:, 0] *= (w0_orig / w_resize)
    mkpts0[:, 1] *= (h0_orig / h_resize)
    mkpts0_rotated_vis = mkpts0.copy()
    if abs(source_roll_deg) > 1e-6:
        mkpts0 = rotate_points_about_image_center(mkpts0.astype(np.float32), -source_roll_deg, (w0_orig, h0_orig))

    mkpts1[:, 0] *= (w1_orig / w_resize)
    mkpts1[:, 1] *= (h1_orig / h_resize)
    mkpts1_rotated_vis = mkpts1.copy()

    if mkpts0.size > 0:
        mkpts0[:, 0] = np.clip(mkpts0[:, 0], 0, w0_orig - 1)
        mkpts0[:, 1] = np.clip(mkpts0[:, 1], 0, h0_orig - 1)
    if mkpts1.size > 0:
        mkpts1[:, 0] = np.clip(mkpts1[:, 0], 0, w1_orig - 1)
        mkpts1[:, 1] = np.clip(mkpts1[:, 1], 0, h1_orig - 1)
    if mkpts0_rotated_vis.size > 0:
        mkpts0_rotated_vis[:, 0] = np.clip(mkpts0_rotated_vis[:, 0], 0, w0_orig - 1)
        mkpts0_rotated_vis[:, 1] = np.clip(mkpts0_rotated_vis[:, 1], 0, h0_orig - 1)
    if mkpts1_rotated_vis.size > 0:
        mkpts1_rotated_vis[:, 0] = np.clip(mkpts1_rotated_vis[:, 0], 0, w1_orig - 1)
        mkpts1_rotated_vis[:, 1] = np.clip(mkpts1_rotated_vis[:, 1], 0, h1_orig - 1)

    matches = np.stack([mkpts0, mkpts1], axis=1).astype(np.uint32)

    if write_out_flag:
        source_image, target_image = Path(img0_path), Path(img1_path)
        pair_name = f'{source_image.parents[1].name}_to_{target_image.parents[1].name}'
        debug_dir = ROOT / 'outputs/matching_debug' / source_image.parents[2].name / pair_name
        debug_dir.mkdir(parents=True, exist_ok=True)
        color = cm.jet(mconf, alpha=0.7)
        text = [
            'LoFTR',
            'Matches: {}'.format(len(mkpts0)),
            'Applied roll: {:.2f} deg'.format(source_roll_deg),
        ]
        img0_parts = os.path.normpath(img0_path).split(os.sep)
        img1_parts = os.path.normpath(img1_path).split(os.sep)
        min_len = min(len(img0_parts), len(img1_parts))
        split_idx = 0
        for i in range(min_len):
            if img0_parts[i] != img1_parts[i]:
                split_idx = i
                break
        else:
            split_idx = min_len

        folder_tag = img0_parts[split_idx - 1] + "_" + img1_parts[split_idx - 1]
        sub_path0 = "_".join(img0_parts[split_idx:-1])
        sub_path1 = "_".join(img1_parts[split_idx:-1])
        name0 = os.path.splitext(os.path.basename(img0_path))[0]
        name1 = os.path.splitext(os.path.basename(img1_path))[0]
        debug_name_base = f"{folder_tag}_{sub_path0}_{name0}__{sub_path1}_{name1}"
        make_matching_figure(
            img0_rgb,
            img1_rgb,
            mkpts0_rotated_vis,
            mkpts1_rotated_vis,
            color,
            mkpts0_rotated_vis,
            mkpts1_rotated_vis,
            text,
            path=(debug_dir / f"{debug_name_base}_rotated_input.png").as_posix()
        )
        make_matching_figure(
            img0_rgb_unrotated,
            img1_rgb,
            mkpts0,
            mkpts1,
            color,
            mkpts0,
            mkpts1,
            text,
            path=(debug_dir / f"{debug_name_base}_unrotated_output.png").as_posix()
        )
    return matches, mconf


def filter_and_collect_pose_pairs(
    npz1_path: str,
    images1_folder: str,
    masks1_folder: str,
    npz2_path: str,
    images2_folder: str,
    masks2_folder: str,
    delta_RT: torch.tensor,
    K: int = 15,
    file_ext: str = ".png"
):
    if isinstance(delta_RT, torch.Tensor):
        delta_RT_np = delta_RT.detach().cpu().numpy().astype(np.float32)
    else:
        delta_RT_np = np.asarray(delta_RT, dtype=np.float32)

    if K == 0:
        combined_pairs = find_near_camera_poses(npz1_path, npz2_path, delta_RT_np)
    else:
        combined_pairs = find_K_near_camera_poses(npz1_path, npz2_path, delta_RT_np, K=K)
        print(combined_pairs)

    poses1 = from_npz_get_poses(npz1_path)
    poses2 = from_npz_get_poses(npz2_path)
    results = []
    for (i, j) in combined_pairs:
        img1_path = os.path.join(images1_folder, f"{i:03}{file_ext}")
        mask1_path = os.path.join(masks1_folder, f"{i:03}{file_ext}")
        img2_path = os.path.join(images2_folder, f"{j:03}{file_ext}")
        mask2_path = os.path.join(masks2_folder, f"{j:03}{file_ext}")
        source_pose_delta = delta_RT_np @ poses1[i]
        pose_roll_deg = normalize_angle_deg(estimate_relative_roll_deg(source_pose_delta, poses2[j]))
        applied_roll_deg = normalize_angle_deg(-pose_roll_deg)
        results.append({
            "idx1": i,
            "idx2": j,
            "img1": img1_path,
            "mask1": mask1_path,
            "img2": img2_path,
            "mask": mask2_path,
            "mask2": mask2_path,
            "source_roll_deg": applied_roll_deg,
            "pose_estimated_roll_deg": pose_roll_deg,
            "applied_source_roll_deg": applied_roll_deg,
            "roll_strategy": "negate_pose_estimate"
        })

    write_roll_debug_json(
        results=results,
        delta_RT=delta_RT_np,
        npz1_path=npz1_path,
        npz2_path=npz2_path,
        verbose=False
    )
    return results


# -------------------- 以下是一个简单的使用示例 --------------------
if __name__ == "__main__":
    raise SystemExit("Use pipeline.py register --case CASE_ID")

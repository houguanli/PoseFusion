import os, sys, copy, glob, json, time, random, argparse, cv2, re
from shutil import copyfile
from project_paths import ROOT, load_registration_config, resolve_network_config
from tqdm import tqdm, trange
import math
import mmengine
import imageio
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import trimesh
from pathlib import Path
from lib import utils, dtu_eval
# from torch.utils.tensorboard import SummaryWriter
from lib.load_data import load_data
from lib.utils import rgb_to_luminance, get_sobel, calc_grad, \
    GradLoss, write_ply, load_point_cloud, get_root_logger
from collections import defaultdict
import lib.dvgo_ori as dvgo_ori
import lib.voxurf_fine as Model
import robust_loss_pytorch as robust

from torch_efficient_distloss import flatten_eff_distloss
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from run import load_everything
from obb_reg_helper import *
from image_feature_helper import *
from LoFTR_helper import *


def atomic_write_json(json_path, data):
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for key, value in data.items():
        if isinstance(key, str) and key.startswith("__"):
            continue
        payload[key] = data.get('__path_fields__', {}).get(key, value)
    tmp_path = json_path.with_suffix(json_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    os.replace(tmp_path, json_path)


def pose_translation_distance(M1: np.ndarray, M2: np.ndarray) -> float:
    c1 = np.asarray(M1[:3, 3], dtype=np.float32)
    c2 = np.asarray(M2[:3, 3], dtype=np.float32)
    return float(np.linalg.norm(c1 - c2))


def prepare_for_cv_safe(img):
    # 确保 tensor → cpu → numpy
    if isinstance(img, torch.Tensor):
        img = img.clone().detach().cpu().numpy()
    else:
        img = np.array(img)

    # 转成 uint8
    img = (img * 255).clip(0, 255).astype(np.uint8) if img.dtype != np.uint8 else img
    # RGB → BGR
    if img.shape[-1] == 3:
        img = img[..., ::-1]

    return img


def _rect_faces(rect):
    vertices = generate_full_rect_using_rect(rect)
    return [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
    ]


def _sample_quad(face, steps=24):
    face = np.asarray(face, dtype=np.float32)
    u = np.linspace(0.05, 0.95, steps, dtype=np.float32)
    v = np.linspace(0.05, 0.95, steps, dtype=np.float32)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    points = (
        (1.0 - uu[..., None]) * (1.0 - vv[..., None]) * face[0]
        + uu[..., None] * (1.0 - vv[..., None]) * face[1]
        + uu[..., None] * vv[..., None] * face[2]
        + (1.0 - uu[..., None]) * vv[..., None] * face[3]
    )
    return points.reshape(-1, 3)


def _project_points_to_image(points, pose, K, height, width):
    if isinstance(pose, torch.Tensor):
        pose = pose.detach().cpu().numpy()
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    points = np.asarray(points, dtype=np.float32)
    w2c = np.linalg.inv(np.asarray(pose, dtype=np.float32))
    pts_h = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    cam = (pts_h @ w2c.T)[:, :3]
    z = cam[:, 2]
    # NeuS decomposition returns OpenCV c2w (+Z forward, +Y down).
    front = z > 1e-5
    u = K[0, 0] * cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    valid = front & (u >= 0) & (u < width - 1) & (v >= 0) & (v < height - 1)
    return u[valid].astype(np.int32), v[valid].astype(np.int32), valid


def _colors_to_hsv_hist(colors, h_bins=36, s_bins=16, v_bins=16):
    colors = np.asarray(colors)
    if colors.size == 0:
        return None
    if colors.dtype != np.uint8:
        if colors.max() <= 1.5:
            colors = colors * 255.0
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    rgb = colors.reshape(-1, 1, 3)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).reshape(-1, 3)
    weights = (hsv[:, 2] > 12).astype(np.float32) * (0.25 + 0.75 * hsv[:, 1].astype(np.float32) / 255.0)
    if float(weights.sum()) < 1e-6:
        return None
    h_hist, _ = np.histogram(hsv[:, 0], bins=h_bins, range=(0, 180), weights=weights)
    s_hist, _ = np.histogram(hsv[:, 1], bins=s_bins, range=(0, 256), weights=weights)
    v_hist, _ = np.histogram(hsv[:, 2], bins=v_bins, range=(0, 256), weights=weights)
    hist = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    hist /= float(hist.sum() + 1e-8)
    return hist


def _hist_chi_square(h1, h2):
    return float(0.5 * np.sum(((h1 - h2) ** 2) / (h1 + h2 + 1e-8)))


def _look_at_c2w(camera_pos, target, up_hint=None, forward_is_positive_z=False):
    camera_pos = np.asarray(camera_pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = target - camera_pos
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-8:
        return None
    forward = forward / forward_norm
    z_axis = forward if forward_is_positive_z else -forward
    if up_hint is None:
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    up_hint = np.asarray(up_hint, dtype=np.float64)
    if abs(np.dot(up_hint / (np.linalg.norm(up_hint) + 1e-8), z_axis)) > 0.92:
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = np.cross(up_hint, z_axis)
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        return None
    x_axis = x_axis / x_norm
    y_axis = np.cross(z_axis, x_axis)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = x_axis.astype(np.float32)
    c2w[:3, 1] = y_axis.astype(np.float32)
    c2w[:3, 2] = z_axis.astype(np.float32)
    c2w[:3, 3] = camera_pos.astype(np.float32)
    return c2w


def _render_hsv_signature(image, bins=(36, 16, 16)):
    image = np.asarray(image)
    if image.size == 0:
        return None
    rgb = image[..., :3].reshape(-1, 3)
    if rgb.max() <= 1.5:
        rgb_u8 = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    else:
        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    # Ignore black background and very dark empty render areas.
    keep = np.max(rgb_u8, axis=1) > 12
    if int(np.count_nonzero(keep)) < 64:
        return None
    return _colors_to_hsv_hist(rgb_u8[keep], h_bins=bins[0], s_bins=bins[1], v_bins=bins[2])


def _is_near_identity(rt, rot_tol_deg=5.0, trans_tol=0.02):
    rt = np.asarray(rt, dtype=np.float64)
    rot = rt[:3, :3]
    cos = np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0)
    rot_deg = float(np.degrees(np.arccos(cos)))
    trans = float(np.linalg.norm(rt[:3, 3]))
    return rot_deg <= rot_tol_deg and trans <= trans_tol
def get_optimizer(mode, honkaiStart):
    optimizer = None
    if  mode == "refine_rt":
        optimizer = torch.optim.Adam(
            [
                {'params': getattr(honkaiStart, 'raw_translation'), 'lr': 1e-4},
                {'params': getattr(honkaiStart, 'raw_quaternion'), 'lr': 1e-4},
                {'params': list(getattr(honkaiStart, 'adaptive_loss').parameters()), 'lr': 1e-2}
            ],
            amsgrad=False
        )
    return optimizer

def rotation_matrix_to_quaternion(R):
    """
    Convert a rotation matrix into a quaternion.
    Parameters:
    - R: A 3x3 rotation matrix.
    Returns:
    - q: A quaternion [qw, qx, qy, qz] as a numpy array.
    """
    # Ensure the matrix is numpy array for calculations
    R = np.asarray(R)
    # Allocate space for the quaternion
    q = np.empty((4,))
    # Compute the trace of the matrix
    tr = R.trace()
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2  # S=4*qw
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (R[0, 1] + R[1, 0]) / S
        q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
        q[0] = (R[0, 2] - R[2, 0]) / S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
        q[0] = (R[1, 0] - R[0, 1]) / S
        q[1] = (R[0, 2] + R[2, 0]) / S
        q[2] = (R[1, 2] + R[2, 1]) / S
        q[3] = 0.25 * S
    return q


def transfer_points_to_local_axis(points, quaternion, translation, device="cuda"):
    # convert to 4X4 mat
    w, x, y, z = quaternion
    rotate_mat = torch.zeros((3, 3), device=device)
    rotate_mat[0, 0] = 1 - 2 * (y ** 2 + z ** 2)
    rotate_mat[0, 1] = 2 * (x * y - z * w)
    rotate_mat[0, 2] = 2 * (x * z + y * w)
    rotate_mat[1, 0] = 2 * (x * y + z * w)
    rotate_mat[1, 1] = 1 - 2 * (x ** 2 + z ** 2)
    rotate_mat[1, 2] = 2 * (y * z - x * w)
    rotate_mat[2, 0] = 2 * (x * z - y * w)
    rotate_mat[2, 1] = 2 * (y * z + x * w)
    rotate_mat[2, 2] = 1 - 2 * (x ** 2 + y ** 2)
    transform_matrix = torch.zeros((4, 4), device=device)
    transform_matrix[0:3, 0:3] = rotate_mat
    transform_matrix[0:3, 3] = translation
    transform_matrix[3, 3] = 1.0
    ones_row = torch.ones(1, points.shape[0], dtype=points.dtype, device=points.device)
    # import pdb; pdb.set_trace()
    points_extended = torch.cat((points.T, ones_row), dim=0)
    transfered_points = transform_matrix @ points_extended
    points_3d = transfered_points[:-1, :]
    return points_3d.T


def decompose_RT(input_mat, flag = "quad"):
    # assume input mat is 4x4 mat
    r = input_mat[:3, :3]
    t = input_mat[:3, 3]

    if flag == "quad": # decompose mat to quad and T
        r = rotation_matrix_to_quaternion(r)
    return r, t


def quaternion_to_rotation_matrix_torch(quaternion, device="cuda"):
    w, x, y, z = quaternion
    rotate_mat = torch.zeros((3, 3), dtype=quaternion.dtype, device=device)
    rotate_mat[0, 0] = 1 - 2 * (y ** 2 + z ** 2)
    rotate_mat[0, 1] = 2 * (x * y - z * w)
    rotate_mat[0, 2] = 2 * (x * z + y * w)
    rotate_mat[1, 0] = 2 * (x * y + z * w)
    rotate_mat[1, 1] = 1 - 2 * (x ** 2 + z ** 2)
    rotate_mat[1, 2] = 2 * (y * z - x * w)
    rotate_mat[2, 0] = 2 * (x * z - y * w)
    rotate_mat[2, 1] = 2 * (y * z + x * w)
    rotate_mat[2, 2] = 1 - 2 * (x ** 2 + y ** 2)
    return rotate_mat

def seed_everything():
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

def config_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default=None, help='config file')
    return parser

class  MPReg(nn.Module):
    def _scene_family_name(self, scene_name):
        return re.sub(r'_pose\d+$', '', scene_name)

    def _coarse_case_name(self):
        if self.config_path is not None:
            return self.config_path.stem
        return f"{self._scene_family_name(self.source_name)}_to_{self._scene_family_name(self.target_name)}"

    def _coarse_debug_dir(self):
        return ROOT / 'outputs/registration_debug' / self._scene_family_name(self.source_name) / self._coarse_case_name() / 'coarse'

    def _serialize_for_json(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return value.as_posix()
        if isinstance(value, dict):
            return {k: self._serialize_for_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._serialize_for_json(v) for v in value]
        return value

    def _crop_mesh_for_coarse(self, mesh, min_z=None, max_z=None):
        cropped = mesh.copy()
        if len(cropped.faces) == 0:
            return cropped
        vertices = np.asarray(cropped.vertices)
        faces = np.asarray(cropped.faces)
        face_vertices = vertices[faces]
        if max_z is not None:
            keep_faces = np.all(face_vertices[:, :, 2] < max_z, axis=1)
        else:
            keep_faces = np.all(face_vertices[:, :, 2] >= min_z, axis=1)
        cropped.update_faces(keep_faces)
        cropped.remove_unreferenced_vertices()
        return cropped

    def _prepare_coarse_mesh(self, mesh_path, role):
        mesh = trimesh.load_mesh(mesh_path)
        coarse_mesh = mesh.copy()
        coarse_mesh_path = Path(mesh_path)
        min_z = self.config.get('coarse_mesh_min_z', None)
        max_z = self.config.get('coarse_mesh_max_z', None)
        if max_z is not None:
            coarse_mesh = self._crop_mesh_for_coarse(coarse_mesh, max_z=float(max_z))
            if len(coarse_mesh.faces) == 0:
                print(
                    f"WARNING: coarse mesh z<{max_z} crop removed all faces for {role}: "
                    f"{mesh_path}; using uncropped mesh for coarse registration"
                )
                coarse_mesh = mesh.copy()
                return coarse_mesh_path.as_posix(), coarse_mesh
            threshold_tag = str(max_z).replace('-', 'm').replace('.', 'p')
            coarse_debug_dir = self._coarse_debug_dir()
            coarse_debug_dir.mkdir(parents=True, exist_ok=True)
            coarse_mesh_path = coarse_debug_dir / f"{role}_mesh_z_lt_{threshold_tag}.ply"
            coarse_mesh.export(coarse_mesh_path.as_posix())
        elif min_z is not None:
            coarse_mesh = self._crop_mesh_for_coarse(coarse_mesh, min_z=float(min_z))
            if len(coarse_mesh.faces) == 0:
                print(
                    f"WARNING: coarse mesh z>={min_z} crop removed all faces for {role}: "
                    f"{mesh_path}; using uncropped mesh for coarse registration"
                )
                coarse_mesh = mesh.copy()
                return coarse_mesh_path.as_posix(), coarse_mesh
            threshold_tag = str(min_z).replace('-', 'm').replace('.', 'p')
            coarse_debug_dir = self._coarse_debug_dir()
            coarse_debug_dir.mkdir(parents=True, exist_ok=True)
            coarse_mesh_path = coarse_debug_dir / f"{role}_mesh_z_ge_{threshold_tag}.ply"
            coarse_mesh.export(coarse_mesh_path.as_posix())
        return coarse_mesh_path.as_posix(), coarse_mesh

    def _compute_coarse_obb(self, mesh_path, mesh):
        obb_mode = self.config.get('coarse_obb_mode', 'pca')
        if obb_mode == 'min_volume':
            try:
                corners = np.asarray(min_volume_obb_corners_from_mesh(mesh=mesh), dtype=np.float64)
                if corners.shape == (8, 3) and np.isfinite(corners).all():
                    extents = corners.max(axis=0) - corners.min(axis=0)
                    if np.count_nonzero(extents > 1e-9) >= 2:
                        return corners
                print(f"WARNING: invalid min_volume OBB corners for {mesh_path}; using AABB fallback")
            except Exception as exc:
                print(f"WARNING: min_volume OBB failed for {mesh_path}: {exc}; using AABB fallback")
            bounds = np.asarray(mesh.bounds, dtype=np.float64)
            mn, mx = bounds[0].copy(), bounds[1].copy()
            span = mx - mn
            for axis in range(3):
                if span[axis] <= 1e-9:
                    mn[axis] -= 5e-4
                    mx[axis] += 5e-4
            return np.array(
                [
                    [mn[0], mn[1], mn[2]],
                    [mx[0], mn[1], mn[2]],
                    [mx[0], mx[1], mn[2]],
                    [mn[0], mx[1], mn[2]],
                    [mn[0], mn[1], mx[2]],
                    [mx[0], mn[1], mx[2]],
                    [mx[0], mx[1], mx[2]],
                    [mn[0], mx[1], mx[2]],
                ],
                dtype=np.float64,
            )
        return Mesh2SDF(mesh_path=mesh_path).calculate_obb_from_point_cloud()

    def _face_color_signatures(self, rect, data_dict, max_views=12, steps=24):
        images = data_dict["images"]
        poses = data_dict["poses"]
        Ks = data_dict["Ks"]
        height = int(data_dict["H"])
        width = int(data_dict["W"])
        if isinstance(images, torch.Tensor):
            images_np = images.detach().cpu().numpy()
        else:
            images_np = np.asarray(images)
        if images_np.max() > 2.0:
            images_np = images_np.astype(np.float32) / 255.0

        view_count = min(len(images_np), max_views)
        if view_count <= 0:
            return None
        view_indices = np.linspace(0, len(images_np) - 1, view_count, dtype=int)
        signatures = []
        sample_counts = []
        for face in _rect_faces(rect):
            points = _sample_quad(face, steps=steps)
            colors = []
            for view_idx in view_indices:
                u, v, _ = _project_points_to_image(points, poses[view_idx], Ks[view_idx], height, width)
                if len(u) == 0:
                    continue
                pix = images_np[view_idx, v, u, :3]
                silhouette = data_dict['masks'][view_idx]
                if isinstance(silhouette, torch.Tensor):
                    silhouette = silhouette.detach().cpu().numpy()
                keep = (np.max(pix, axis=1) > 0.04) & (silhouette[v, u].reshape(len(u), -1).max(axis=-1) > 0.5)
                if np.any(keep):
                    colors.append(pix[keep])
            if not colors:
                signatures.append(None)
                sample_counts.append(0)
                continue
            colors = np.concatenate(colors, axis=0)
            hist = _colors_to_hsv_hist(colors)
            signatures.append(hist)
            sample_counts.append(int(len(colors)))
        if sum(sig is not None for sig in signatures) < 3:
            return None
        return {"signatures": signatures, "sample_counts": sample_counts}

    def _face_color_difference(self, target_rect, source_rect, target_face_cache=None):
        target_payload = target_face_cache or self._face_color_signatures(target_rect, self.target_data_dict)
        source_payload = self._face_color_signatures(source_rect, self.source_data_dict)
        if target_payload is None or source_payload is None:
            return None
        diffs = []
        records = []
        for face_idx, (target_sig, source_sig) in enumerate(zip(target_payload["signatures"], source_payload["signatures"])):
            if target_sig is None or source_sig is None:
                continue
            diff = _hist_chi_square(target_sig, source_sig)
            diffs.append(diff)
            records.append({
                "face": face_idx,
                "hsv_chi_square": diff,
                "target_samples": target_payload["sample_counts"][face_idx],
                "source_samples": source_payload["sample_counts"][face_idx],
            })
        if len(diffs) < 3:
            return None
        drop_worst_faces = max(0, min(int(self.config.get("coarse_drop_worst_faces", 0)), len(diffs) - 1))
        order = np.argsort(diffs)
        kept = set(int(i) for i in order[:len(diffs) - drop_worst_faces])
        for record_idx, record in enumerate(records):
            record["kept_for_trimmed_mean"] = record_idx in kept
        trimmed_diffs = [diff for idx, diff in enumerate(diffs) if idx in kept]
        return {
            "avg_face_color_diff": float(np.mean(trimmed_diffs)),
            "raw_avg_face_color_diff": float(np.mean(diffs)),
            "drop_worst_faces": int(drop_worst_faces),
            "face_color_records": records,
        }

    def _face_render_camera(self, face, rect_center, distance_scale=2.8, forward_is_positive_z=False):
        face = np.asarray(face, dtype=np.float64)
        face_center = face.mean(axis=0)
        direction = face_center - np.asarray(rect_center, dtype=np.float64)
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            direction = face_center.copy()
            norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return None
        direction = direction / norm
        extent = max(float(np.linalg.norm(face[1] - face[0])), float(np.linalg.norm(face[3] - face[0])), 1e-3)
        camera_pos = face_center + direction * extent * distance_scale
        up_hint = face[3] - face[0]
        return _look_at_c2w(
            camera_pos,
            face_center,
            up_hint=up_hint,
            forward_is_positive_z=forward_is_positive_z,
        )

    def _obb_face_render_signature_difference(self, target_rect, source_rect, save_debug_dir=None, candidate_index=None):
        if not self.config.get("use_obb_face_render_signature", False):
            return None
        resolution = int(self.config.get("obb_face_render_resolution", 128))
        distance_scale = float(self.config.get("obb_face_render_distance_scale", 2.8))
        drop_worst_faces = max(0, min(int(self.config.get("obb_face_render_drop_worst_faces", self.config.get("coarse_drop_worst_faces", 0))), 5))
        skip_min_z = self.config.get("obb_face_render_skip_min_z", self.config.get("coarse_mesh_min_z", None))
        skip_margin = float(self.config.get("obb_face_render_skip_z_margin", 0.005))
        focal = float(self.config.get("obb_face_render_focal", resolution * 1.35))

        target_faces = _rect_faces(target_rect)
        source_faces = _rect_faces(source_rect)
        target_center = np.mean(np.asarray(generate_full_rect_using_rect(target_rect), dtype=np.float64), axis=0)
        source_center = np.mean(np.asarray(generate_full_rect_using_rect(source_rect), dtype=np.float64), axis=0)
        K_np = np.array(
            [[focal, 0.0, resolution * 0.5], [0.0, focal, resolution * 0.5], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        K = torch.from_numpy(K_np).to(device=self.target_data_dict['poses'][0].device, dtype=self.target_data_dict['poses'][0].dtype)
        records = []
        diffs = []

        for face_idx, (target_face, source_face) in enumerate(zip(target_faces, source_faces)):
            target_face = np.asarray(target_face, dtype=np.float64)
            source_face = np.asarray(source_face, dtype=np.float64)
            skipped_reason = None
            if skip_min_z is not None:
                threshold = float(skip_min_z) + skip_margin
                if float(np.max(target_face[:, 2])) < threshold or float(np.max(source_face[:, 2])) < threshold:
                    skipped_reason = "near_z_crop"
            if skipped_reason is not None:
                records.append({"face": face_idx, "skipped": True, "reason": skipped_reason})
                continue

            target_pose_np = self._face_render_camera(
                target_face,
                target_center,
                distance_scale=distance_scale,
                forward_is_positive_z=bool(self.target_net_cfg.data.inverse_y),
            )
            source_pose_np = self._face_render_camera(
                source_face,
                source_center,
                distance_scale=distance_scale,
                forward_is_positive_z=bool(self.source_net_cfg.data.inverse_y),
            )
            if target_pose_np is None or source_pose_np is None:
                records.append({"face": face_idx, "skipped": True, "reason": "invalid_face_camera"})
                continue
            target_pose = torch.from_numpy(target_pose_np).to(device=self.target_data_dict['poses'][0].device, dtype=self.target_data_dict['poses'][0].dtype)
            source_pose = torch.from_numpy(source_pose_np).to(device=self.source_data_dict['poses'][0].device, dtype=self.source_data_dict['poses'][0].dtype)

            target_render = self.render_single_image(
                model=self.target_model, width=resolution, height=resolution, K=K, pose=target_pose,
                cfg=self.target_net_cfg, data_dict=self.target_data_dict, savedir=None
            )
            source_render = self.render_single_image(
                model=self.source_model, width=resolution, height=resolution, K=K, pose=source_pose,
                cfg=self.source_net_cfg, data_dict=self.source_data_dict, savedir=None
            )
            target_sig = _render_hsv_signature(target_render)
            source_sig = _render_hsv_signature(source_render)
            if target_sig is None or source_sig is None:
                records.append({"face": face_idx, "skipped": True, "reason": "empty_render"})
                continue
            diff = _hist_chi_square(target_sig, source_sig)
            record = {"face": face_idx, "skipped": False, "hsv_chi_square": float(diff)}
            records.append(record)
            diffs.append(diff)

            if save_debug_dir is not None and self.config.get("save_obb_face_render_debug", False):
                prefix = Path(save_debug_dir) / f"index{int(candidate_index):02d}_face{face_idx}"
                cv2.imwrite(prefix.as_posix() + "_target_face_render.png", prepare_for_cv_safe(target_render))
                cv2.imwrite(prefix.as_posix() + "_source_face_render.png", prepare_for_cv_safe(source_render))

        if len(diffs) < int(self.config.get("obb_face_render_min_faces", 3)):
            return {"avg_obb_face_render_diff": None, "raw_avg_obb_face_render_diff": None, "records": records}
        order = np.argsort(diffs)
        kept_count = max(1, len(diffs) - min(drop_worst_faces, len(diffs) - 1))
        kept_positions = set(int(i) for i in order[:kept_count])
        kept_diffs = []
        valid_record_pos = 0
        for record in records:
            if record.get("skipped", False):
                continue
            keep = valid_record_pos in kept_positions
            record["kept_for_trimmed_mean"] = keep
            if keep:
                kept_diffs.append(float(record["hsv_chi_square"]))
            valid_record_pos += 1
        return {
            "avg_obb_face_render_diff": float(np.mean(kept_diffs)),
            "raw_avg_obb_face_render_diff": float(np.mean(diffs)),
            "drop_worst_faces": int(drop_worst_faces),
            "records": records,
        }

    def __init__(self, config, device):
        super().__init__()

        self.config = config
        self.config_path = Path(config['__config_path__']) if config.get('__config_path__') else None
        self.device = device
        self.trace_threshold_rotation, self.trace_threshold_translation = self.config['trace_threshold_rotation'], self.config['trace_threshold_translation']
        self.target_net_cfg = resolve_network_config(mmengine.Config.fromfile(config['target_net_config']))
        self.source_net_cfg = resolve_network_config(mmengine.Config.fromfile(config['source_net_config']))
        self.target_net_cfg['data']['datadir'] = config['target_datadir']
        self.source_net_cfg['data']['datadir'] = config['source_datadir']
        # Registration reads actual background RGB; silhouettes gate feature matches.
        self.target_net_cfg.data.mode.update(wmask=False)
        self.source_net_cfg.data.mode.update(wmask=False)
        self.source_name = self.config.get('source_name', os.path.basename(self.config['source_datadir']))
        self.target_name = self.config.get('target_name', os.path.basename(self.config['target_datadir']))
        self.source_coarse_mesh_path, self.source_coarse_mesh = self._prepare_coarse_mesh(self.config['source_mesh_dir'], 'source')
        self.target_coarse_mesh_path, self.target_coarse_mesh = self._prepare_coarse_mesh(self.config['target_mesh_dir'], 'target')
        self.mesh_target_sdf = Mesh2SDF(mesh_path=self.target_coarse_mesh_path)
        self.mesh_source_sdf = Mesh2SDF(mesh_path=self.source_coarse_mesh_path)
         # load mesh and compute the oriented bounding box for coarse registration
        self.target_mesh_obb = self._compute_coarse_obb(self.target_coarse_mesh_path, self.target_coarse_mesh)
        self.source_mesh_obb = self._compute_coarse_obb(self.source_coarse_mesh_path, self.source_coarse_mesh)
        self.target_mesh = trimesh.load_mesh(self.config['target_mesh_dir'])

        self.source_data_dict = load_everything(self.config, self.source_net_cfg)
        self.source_data_dict['Ks'] = torch.from_numpy(self.source_data_dict['Ks']).to(self.device, dtype=torch.float)
        self.source_data_dict['intrinsics_all_inv'] = torch.inverse(self.source_data_dict['Ks'])
        # import pdb; pdb.set_trace()
        self.source_data_dict['H'] = int(self.source_data_dict['HW'][0][0])
        self.source_data_dict['W'] = int(self.source_data_dict['HW'][0][1])

        self.target_data_dict = load_everything(self.config, self.target_net_cfg)
        self.target_data_dict['Ks'] = torch.from_numpy(self.target_data_dict['Ks']).to(self.device, dtype=torch.float)
        self.target_data_dict['intrinsics_all_inv'] = torch.inverse(self.target_data_dict['Ks'])
        self.target_data_dict['H'] = int(self.target_data_dict['HW'][0][0])
        self.target_data_dict['W'] = int(self.target_data_dict['HW'][0][1])
        source_ckpt_path = os.path.join(config['source_exp_dir'], 'surf_last.tar')
        source_kwargs = self.source_net_cfg.surf_model_and_render
        print(">>> Loading Source from {}".format(source_ckpt_path))
        self.source_model = utils.load_model(Model.Voxurf, source_ckpt_path, source_kwargs).to(self.device)

        target_ckpt_path = os.path.join(config['target_exp_dir'], 'surf_last.tar')
        target_kwargs = self.target_net_cfg.surf_model_and_render
        print(">>> Loading Target from {}".format(target_ckpt_path))
        self.target_model = utils.load_model(Model.Voxurf, target_ckpt_path, target_kwargs).to(self.device)

        
        sdf_grid_test = self.get_sdf_grid(self.target_model)
        self.raw_reg_time_sec = 0.0
        raw_reg = None if self.config.get('force_recompute_raw_reg', False) else self.config.get('raw_reg', None)
        if raw_reg is not None:
            OBB_RT = np.asarray(raw_reg, dtype=np.float32)
            best_result = {
                'index': 'config_raw_reg',
                'obb_RT': OBB_RT,
                'sdf_diff': None,
                'avg_feature_diff': None,
                'valid_pose_pairs': [],
            }
            print(">>> Using raw_reg from config as coarse initialization.")
        else:
            print(">>> No raw_reg found in config, running OBB coarse registration.")
            coarse_start = time.perf_counter()
            OBB_RT, best_result = self.OBB_Registration(save_debug=self.config.get('save_coarse_debug', False))
            self.raw_reg_time_sec = time.perf_counter() - coarse_start
            self.config['raw_reg'] = OBB_RT.tolist()
            self.config['raw_reg_info'] = {
                'index': best_result.get('index'),
                'sdf_diff': best_result.get('sdf_diff'),
                'avg_feature_diff': best_result.get('avg_feature_diff'),
                'valid_pose_pairs_count': len(best_result.get('valid_pose_pairs', []))
            }
            self.config['raw_reg_time_sec'] = float(self.raw_reg_time_sec)
            if self.config_path is not None:
                atomic_write_json(self.config_path, self.config)
                print(f">>> Saved raw_reg back to {self.config_path}")
        self.coarse_best_result = best_result
        if self.config.get('coarse_only', False):
            print(">>> coarse_only=true: skipping fine registration.")
            self.coarse_rt = OBB_RT
            return
        self.coarse_rt = OBB_RT
        print("Corresponding record:", best_result)
        if raw_reg is not None:
            print(">>> raw_reg_time_sec = 0.000000 (loaded from config)")
        else:
            print(f">>> raw_reg_time_sec = {self.raw_reg_time_sec:.6f}")
        # prepare init rt for fine reg stage
        raw_quaternion, raw_translation = decompose_RT(OBB_RT)

        self.raw_translation = torch.tensor(raw_translation, dtype=torch.float32, requires_grad=True) # this para is raw from bbox align
        self.raw_quaternion  = torch.tensor(raw_quaternion, dtype=torch.float32, requires_grad=True) #
        self.adaptive_loss = robust.AdaptiveLossFunction(
            num_dims=1,
            float_dtype=torch.float,
            device=torch.device(self.device),
            # Keep the robust-loss shape fixed. Optimizing latent_alpha can
            # numerically leave the valid range on hard real cases and trip
            # robust_loss_pytorch's alpha >= 0 assertion during fine reg.
            alpha_lo=0.5,
            alpha_hi=0.5,
            scale_init=1.0
        )
        # self.zeroset_points_source_pose, self.zeroset_points_target_pose = None, None
        self.zeroset_points_source_pose, self.zeroset_points_target_pose = self.init_corresponding_zeroset_points(delta_RT=OBB_RT, save_debug=True, vis_folder=self._coarse_debug_dir().parent / 'fine')
        self.export_coarse_debug(vis_folder=self._coarse_debug_dir().parent / 'fine')
        print("generate points ", len(self.zeroset_points_target_pose), " ", len(self.zeroset_points_source_pose))
        print("========Sample points generation finished========")

        # import pdb; pdb.set_trace()
        if len(self.zeroset_points_source_pose) == 0 or len(self.zeroset_points_target_pose) == 0:
            print("WARNING: zero corresponding points for fine registration; using raw_reg as fine_rt")
            final_T = np.asarray(OBB_RT, dtype=np.float32)
            self.fine_reg_time_sec = 0.0
        else:
            fine_start = time.perf_counter()
            final_T = self.Fine_Registration()
            self.fine_reg_time_sec = time.perf_counter() - fine_start
        final_R = np.asarray(final_T, dtype=np.float32)[:3, :3]
        final_t = np.asarray(final_T, dtype=np.float32)[:3, 3]
        final_q = rotation_matrix_to_quaternion(final_R).astype(np.float32)
        self.config['fine_rt'] = np.asarray(final_T, dtype=np.float32).tolist()
        self.config['fine_translation'] = final_t.tolist()
        self.config['fine_quaternion'] = final_q.tolist()
        self.config['fine_reg_time_sec'] = float(self.fine_reg_time_sec)
        if self.config_path is not None:
            atomic_write_json(self.config_path, self.config)
        self.final_rt = final_T
        self.export_mesh_debug(vis_folder=self._coarse_debug_dir().parent / 'fine', fine_rt=final_T)
        print("=====ALL REG finished========")
        print(f"raw_reg_time_sec: {self.raw_reg_time_sec:.6f}")
        print(f"fine_reg_time_sec: {self.fine_reg_time_sec:.6f}")
        print(final_T)
        # sdf + image pairs informed reg

    #1 is target, 2 is source
    def OBB_Registration(self, save_debug=False):
        print("=========================== Start OBB Registration =========================")
        if self.config.get("debug_break_coarse", False):
            import pdb; pdb.set_trace()
        coarse_debug_dir = self._coarse_debug_dir()
        if save_debug:
            coarse_debug_dir.mkdir(parents=True, exist_ok=True)
        random_uvs = []
        for _ in range(6):
            u = np.random.rand(10000)
            v = np.random.rand(10000)
            random_uvs.append([u, v])

        rect_target = select_rect_from_full(self.target_mesh_obb)
        rect_source = select_rect_from_full(self.source_mesh_obb)
        fr1 = generate_full_rect_using_rect(rect_target)
        fr2 = generate_full_rect_using_rect(rect_source)
        test_indexes = generate_enum_index(select_rect_from_full(self.source_mesh_obb))
        target_face_color_cache = self._face_color_signatures(rect_target, self.target_data_dict)
        drop_worst_faces = int(self.config.get("coarse_drop_worst_faces", 0))

        # Step1: record all diffs, including sdf diffs and color diffs
        all_results = []
        all_source_poses = self.source_data_dict['poses'].cpu().numpy()
        all_target_poses = self.target_data_dict['poses'].cpu().numpy()
        all_source_images, all_target_images = self.source_data_dict['images'] ,self.target_data_dict['images'] # images for differ calculate
        # r1 is source r2 is target!
        for index, (i2, j2, k2, l2) in enumerate(test_indexes):
            tmp_rec_source = Rect(with_init=True,o=fr2[i2],x=fr2[j2],y=fr2[k2],z=fr2[l2])
            tmp_rec_source_full_rect = generate_full_rect_using_rect(tmp_rec_source)
            obb_RT, sdf_diff, face_sdf_result = pairwise_obb_single(
                tmp_rec_source_full_rect,
                fr1,
                random_uvs,
                self.mesh_source_sdf,
                self.mesh_target_sdf,
                drop_worst_faces=drop_worst_faces,
                return_facewise=True,
            )
            face_color_result = self._face_color_difference(
                rect_target,
                tmp_rec_source,
                target_face_cache=target_face_color_cache,
            )
            face_render_result = self._obb_face_render_signature_difference(
                rect_target,
                tmp_rec_source,
                save_debug_dir=coarse_debug_dir if save_debug else None,
                candidate_index=index,
            )
            # Step2_1: apply obb_RT to all source poses
            transformed_source_poses = [obb_RT @ pose for pose in all_source_poses]
            # Step2_2: select candidate pose pairs.
            # For coarse image matching we intentionally do not gate by optical axis:
            # first keep cameras that are spatially close after applying OBB RT,
            # then let the bidirectional rendering similarity handle view-direction effects.
            diff_records = []  # 记录所有符合条件的pair和对应的diff
            diff_records_all = []
            for i, target_pose in enumerate(all_target_poses):
                for j, transformed_source_pose in enumerate(transformed_source_poses):
                    _, diff_rotation, _ = evaluate_extrinsic_difference_OBB(
                        target_pose,
                        transformed_source_pose,
                        rotation_threshold_deg=180.0,
                        translation_threshold=1e9,
                        return_all=True
                    )
                    diff_translation = pose_translation_distance(target_pose, transformed_source_pose)
                    is_valid = diff_translation <= self.trace_threshold_translation
                    diff_score = diff_translation
                    diff_records_all.append({
                        'diff_score': diff_score,
                        'target_index': i,
                        'source_index': j,
                        'diff_rotation': diff_rotation,
                        'diff_translation': diff_translation
                    })
                    if is_valid:
                        diff_records.append({
                            'target_index': i,
                            'source_index': j,
                            'diff_rotation': diff_rotation,
                            'diff_translation': diff_translation
                        })

            print(f"index {index:02d} | Found {len(diff_records)} valid pose pairs before selection")

            # Step2: 如果超过最大k个pair, 则基于加权diff选择前k个
            max_k = 6

            if len(diff_records) > max_k:
                rot_diffs = np.array([r['diff_rotation'] for r in diff_records])
                trans_diffs = np.array([r['diff_translation'] for r in diff_records])

                # Step2.1: 归一化
                norm_rot = (rot_diffs - rot_diffs.min()) / (rot_diffs.max() - rot_diffs.min() + 1e-8)
                norm_trans = (trans_diffs - trans_diffs.min()) / (trans_diffs.max() - trans_diffs.min() + 1e-8)
                # Step2.2: 计算加权总 diff
                final_diff = 0.8 * norm_rot + 0.3 * norm_trans
                # Step2.3: 按照 final_diff 排序取前 k
                sorted_indices = np.argsort(final_diff)
                selected_indices = sorted_indices[:max_k]
                valid_pose_pairs = [diff_records[idx] for idx in selected_indices]
            else:
                # 不足k个则全部保留
                valid_pose_pairs = diff_records
                if len(diff_records) == 0:
                    # 0则取diff最小的k个
                    print("NO matching pairs within threshold!! Adding additional pairs in other standard!" )
                    diff_records_all.sort(key=lambda x: x['diff_score'])
                    valid_pose_pairs = diff_records_all[:(int)(max_k/2)]


            print(f"index {index:02d} | Selected {len(valid_pose_pairs)} pose pairs after applying max_k={max_k}")
            total_feature_diffs = []
            self_device = self.target_data_dict['poses'][0].device
            self_dtype  = self.target_data_dict['poses'][0].dtype

            obb_RT_torch = torch.from_numpy(obb_RT).to(device=self_device, dtype=self_dtype)
            obb_RT_inv_torch = torch.from_numpy(np.linalg.inv(obb_RT)).to(device=self_device, dtype=self_dtype)

            if len(valid_pose_pairs) == 0:
                print(f"index {index:02d} | WARNING: No valid pose pair found, injecting dummy diff = 1.0")
                total_feature_diffs = [1.0]
            else:
                def scale_K(K, scale=0.5):
                    K_new = K.clone() if isinstance(K, torch.Tensor) else K.copy()
                    K_new[0, 0] *= scale
                    K_new[1, 1] *= scale
                    K_new[0, 2] *= scale
                    K_new[1, 2] *= scale
                    return K_new

                scale_factor = 0.5

                for pair in valid_pose_pairs:
                    i = pair['target_index']
                    j = pair['source_index']
                    print(f"    Target Pose #{i}, Source Pose #{j}")

                    pose_target = self.target_data_dict['poses'][i]
                    pose_source = self.source_data_dict['poses'][j]
                    target_w = max(1, int(round(self.target_data_dict['W'] * scale_factor)))
                    target_h = max(1, int(round(self.target_data_dict['H'] * scale_factor)))
                    source_w = max(1, int(round(self.source_data_dict['W'] * scale_factor)))
                    source_h = max(1, int(round(self.source_data_dict['H'] * scale_factor)))

                    # === Target Branch ===
                    pose_source_in_target = obb_RT_inv_torch @ pose_target
                    render_source_in_target = self.render_single_image(
                        model=self.source_model, width=target_w, height=target_h,
                        K=scale_K(self.target_data_dict['Ks'][i], scale_factor),
                        pose=pose_source_in_target, cfg=self.source_net_cfg,
                        data_dict=self.source_data_dict, savedir=None
                    )

                    # === Source Branch ===
                    pose_target_in_source = obb_RT_torch @ pose_source
                    render_target_in_source = self.render_single_image(
                        model=self.target_model, width=source_w, height=source_h,
                        K=scale_K(self.source_data_dict['Ks'][j], scale_factor),
                        pose=pose_target_in_source, cfg=self.target_net_cfg,
                        data_dict=self.target_data_dict, savedir=None
                    )

                    # === Self-source Rendering ===
                    render_source_self = self.render_single_image(
                        model=self.source_model, width=source_w, height=source_h,
                        K=scale_K(self.source_data_dict['Ks'][j], scale_factor),
                        pose=pose_source, cfg=self.source_net_cfg,
                        data_dict=self.source_data_dict, savedir=None
                    )

                    # === Self-target Rendering ===
                    render_target_self = self.render_single_image(
                        model=self.target_model, width=target_w, height=target_h,
                        K=scale_K(self.target_data_dict['Ks'][i], scale_factor),
                        pose=pose_target, cfg=self.target_net_cfg,
                        data_dict=self.target_data_dict, savedir=None
                    )
                    # === Diff ===
                    target_branch_diff = color_feature_difference(render_source_in_target, render_target_self,
                                                                  device=self.device)
                    source_branch_diff = color_feature_difference(render_target_in_source, render_source_self,
                                                                  device=self.device)
                    total_feature_diffs.append((target_branch_diff + source_branch_diff) * 0.5)
                    # === to numpy ===
                    render_source_in_target = prepare_for_cv_safe(render_source_in_target)
                    render_target_in_source = prepare_for_cv_safe(render_target_in_source)
                    render_source_self = prepare_for_cv_safe(render_source_self)
                    render_target_self = prepare_for_cv_safe(render_target_self)
                    source_image = prepare_for_cv_safe(all_source_images[j])
                    target_image = prepare_for_cv_safe(all_target_images[i])
                    # === Debug Save ===
                    if save_debug:
                        prefix = coarse_debug_dir / f'index{index:02d}_pair{i}_{j}'
                        cv2.imwrite((prefix.as_posix() + '_source_image.png'), source_image)
                        cv2.imwrite((prefix.as_posix() + '_target_image.png'), target_image)
                        cv2.imwrite((prefix.as_posix() + '_render_source_in_target.png'), render_source_in_target)
                        cv2.imwrite((prefix.as_posix() + '_render_target_in_source.png'), render_target_in_source)
                        cv2.imwrite((prefix.as_posix() + '_render_source_self.png'), render_source_self)
                        cv2.imwrite((prefix.as_posix() + '_render_target_self.png'), render_target_self)
            print(f">>> For index {index}, Average Feature Diff across valid pairs = {np.mean(total_feature_diffs):.6f}")
            # Step3: Record into all_results
            all_results.append({
                'index': index,
                'obb_RT': obb_RT,
                'sdf_diff': sdf_diff,
                'raw_sdf_diff': face_sdf_result['raw_sdf_diff'],
                'face_sdf_diffs': face_sdf_result['face_sdf_diffs'],
                'sdf_kept_face_indices': face_sdf_result['kept_face_indices'],
                'sdf_drop_worst_faces': face_sdf_result['drop_worst_faces'],
                'avg_feature_diff': np.mean(total_feature_diffs) if len(total_feature_diffs) > 0 else 1e5,
                'avg_face_color_diff': None if face_color_result is None else face_color_result['avg_face_color_diff'],
                'face_color_records': [] if face_color_result is None else face_color_result['face_color_records'],
                'avg_obb_face_render_diff': None if face_render_result is None else face_render_result['avg_obb_face_render_diff'],
                'raw_avg_obb_face_render_diff': None if face_render_result is None else face_render_result['raw_avg_obb_face_render_diff'],
                'obb_face_render_records': [] if face_render_result is None else face_render_result['records'],
                'valid_pose_pairs': valid_pose_pairs # 如果没有有效pair给一个大值
            })
            print(f"Corresponding record:\n{all_results[-1]}")

        # import pdb; pdb.set_trace()
        # Step2: normalization
        sdf_diffs = np.array([r['sdf_diff'] for r in all_results])
        raw_sdf_diffs = np.array([r.get('raw_sdf_diff', r['sdf_diff']) for r in all_results])
        feature_diffs = np.array([r['avg_feature_diff'] for r in all_results])

        norm_sdf = (sdf_diffs - sdf_diffs.min()) / (sdf_diffs.max() - sdf_diffs.min() + 1e-8)
        norm_feature = (feature_diffs - feature_diffs.min()) / (feature_diffs.max() - feature_diffs.min() + 1e-8)
        face_color_values = np.array([
            r['avg_face_color_diff'] if r.get('avg_face_color_diff') is not None else np.nan
            for r in all_results
        ], dtype=np.float64)
        face_render_values = np.array([
            r['avg_obb_face_render_diff'] if r.get('avg_obb_face_render_diff') is not None else np.nan
            for r in all_results
        ], dtype=np.float64)
        norm_face_color = np.full_like(norm_sdf, np.nan)
        norm_face_render = np.full_like(norm_sdf, np.nan)
        if np.isfinite(face_render_values).sum() >= 2:
            finite_face_render = face_render_values[np.isfinite(face_render_values)]
            fill_value = finite_face_render.max() + 1.0
            face_render_for_norm = np.where(np.isfinite(face_render_values), face_render_values, fill_value)
            norm_face_render = (
                (face_render_for_norm - face_render_for_norm.min())
                / (face_render_for_norm.max() - face_render_for_norm.min() + 1e-8)
            )
            # Rubic/dice-style symmetric OBB cases need SDF to be a weak sanity term:
            # use trimmed SDF values, but normalize by the raw untrimmed SDF range so
            # dropped clipped faces do not dominate the candidate ranking.
            norm_sdf_for_face_render = (
                (sdf_diffs - raw_sdf_diffs.min())
                / (raw_sdf_diffs.max() - raw_sdf_diffs.min() + 1e-8)
            )
            norm_sdf_for_face_render = np.clip(norm_sdf_for_face_render, 0.0, 1.0)
            norm_sdf = norm_sdf_for_face_render
            final_diff = 0.20 * norm_sdf + 0.20 * norm_feature + 0.60 * norm_face_render
        elif np.isfinite(face_color_values).sum() >= 2:
            finite_face = face_color_values[np.isfinite(face_color_values)]
            fill_value = finite_face.max() + 1.0
            face_color_for_norm = np.where(np.isfinite(face_color_values), face_color_values, fill_value)
            norm_face_color = (
                (face_color_for_norm - face_color_for_norm.min())
                / (face_color_for_norm.max() - face_color_for_norm.min() + 1e-8)
            )
            final_diff = 0.35 * norm_sdf + 0.25 * norm_feature + 0.40 * norm_face_color
        else:
            final_diff = 0.6 * norm_sdf + 0.4 * norm_feature

        if self.config.get('ignore_identity_coarse', False):
            for idx, result in enumerate(all_results):
                if _is_near_identity(
                    result['obb_RT'],
                    float(self.config.get('identity_rot_tol_deg', 5.0)),
                    float(self.config.get('identity_trans_tol', 0.02)),
                ):
                    final_diff[idx] = float('inf')

        for idx, score in enumerate(final_diff):
            all_results[idx]['final_diff'] = float(score)
            if np.isfinite(norm_face_color[idx]):
                all_results[idx]['norm_face_color'] = float(norm_face_color[idx])
            if np.isfinite(norm_face_render[idx]):
                all_results[idx]['norm_obb_face_render'] = float(norm_face_render[idx])

        best_index = np.argmin(final_diff)
        best_result = all_results[best_index]
        if save_debug:
            debug_payload = {
                'source_name': self.source_name,
                'target_name': self.target_name,
                'source_mesh_input': self.config['source_mesh_dir'],
                'target_mesh_input': self.config['target_mesh_dir'],
                'source_coarse_mesh_path': self.source_coarse_mesh_path,
                'target_coarse_mesh_path': self.target_coarse_mesh_path,
                'coarse_obb_mode': self.config.get('coarse_obb_mode', 'pca'),
                'coarse_mesh_min_z': self.config.get('coarse_mesh_min_z'),
                'best_index': int(best_index),
                'results': self._serialize_for_json(all_results),
            }
            with open(coarse_debug_dir / 'coarse_candidates.json', 'w', encoding='utf-8') as f:
                json.dump(debug_payload, f, indent=2, ensure_ascii=False)
        print("===================== Finish OBB Registration ==================================")
        print(f">>> Best Index: {best_index}, with combined diff: {final_diff[best_index]:.6f}")
        print(f"Corresponding record:\n{all_results[best_index]}")
        return best_result['obb_RT'], best_result


    def init_corresponding_zeroset_points(self, delta_RT, save_debug=False, vis_folder=None):
        with torch.no_grad():
            if save_debug and vis_folder is not None:
                debug_path = vis_folder  # vis_folder is a path item
                out_path_source = str(debug_path) + "/" + self.source_name + ".ply"  # apply rt
                out_path_target = str(debug_path) + "/" + self.target_name + ".ply"
                if not debug_path.exists():
                    os.makedirs(debug_path)
            else:
                out_path_source, out_path_target = None, None

            source_root = self.config['source_datadir']
            target_root = self.config['target_datadir']

            npz_source_path = os.path.join(source_root, 'cameras_sphere.npz')
            images_source_folder = os.path.join(source_root, 'image')
            masks_source_folder = os.path.join(source_root, 'mask')

            npz_target_path = os.path.join(target_root, 'cameras_sphere.npz')
            images_target_folder = os.path.join(target_root, 'image')
            masks_target_folder = os.path.join(target_root, 'mask')

            pairs_info = filter_and_collect_pose_pairs(
                npz1_path=npz_source_path,
                images1_folder=images_source_folder,
                masks1_folder=masks_source_folder,
                npz2_path=npz_target_path,
                images2_folder=images_target_folder,
                masks2_folder=masks_target_folder,
                delta_RT=delta_RT,
                K=5,
                file_ext=".png"
            )  # get valid image pairs， find k nearest
            rays_o_source_all, rays_d_source_all, rays_o_target_all, rays_d_target_all = [], [], [], []
            # ---------------------- Part 2: LoFTR Inference on Each Pair ----------------------
            # For each matched pair, run LoFTR inference to get matches/confidences.
            rays_o_source_all, rays_d_source_all, rays_o_target_all, rays_d_target_all = [], [], [], []
            for pair in pairs_info:
                img_source_path = pair["img1"]  # source
                img_target_path = pair["img2"]  # target

                # Example LoFTR inference call
                matches, confidences = run_loftr_inference(
                    img0_path=img_source_path,  # source
                    img1_path=img_target_path,  # target
                    mask0_path=pair.get("mask1"),
                    mask1_path=pair.get("mask", pair.get("mask2")),
                    use_eloftr=True,  # or False, as needed
                    write_out_flag=True,  # set True if you want to save any debug images
                    source_roll_deg=pair.get("source_roll_deg", 0.0)
                )

                mask_source_path = pair["mask1"]
                mask_target_path = pair.get("mask", pair.get("mask2"))

                # Store the matching results in the same dictionary (保留原来的key)
                pair["matches"] = matches
                pair["confidences"] = confidences

                mask_source_rgb = cv.imread(mask_source_path, cv.IMREAD_COLOR)
                mask_target_rgb = cv.imread(mask_target_path, cv.IMREAD_COLOR)

                mask_source_bin = (np.any(mask_source_rgb != 0, axis=-1)).astype(np.uint8)
                mask_target_bin = (np.any(mask_target_rgb != 0, axis=-1)).astype(np.uint8)

                x_source = matches[:, 0, 0]
                y_source = matches[:, 0, 1]
                x_target = matches[:, 1, 0]
                y_target = matches[:, 1, 1]

                # Keep only those matches for which both masks have a value of 1 at the matched pixels
                valid_mask = (mask_source_bin[y_source, x_source] == 1) & (mask_target_bin[y_target, x_target] == 1)

                filtered_matches = matches[valid_mask]
                filtered_confidences = confidences[valid_mask]

                # generate rays o and rays d and filter based on the pairs detected
                render_kwargs_source = {
                    'inverse_y': self.source_net_cfg['data'].get('inverse_y', True),
                    'flip_x': self.source_net_cfg['data'].get('flip_x', False),
                    'flip_y': self.source_net_cfg['data'].get('flip_y', False)
                }

                render_kwargs_target = {
                    'inverse_y': self.target_net_cfg['data'].get('inverse_y', True),
                    'flip_x': self.target_net_cfg['data'].get('flip_x', False),
                    'flip_y': self.target_net_cfg['data'].get('flip_y', False)
                }
                # import pdb; pdb.set_trace()
                # === Source ===
                image_source_index = pair['idx1']
                rays_o_source, rays_d_source = Model.get_rays(
                    H=self.source_data_dict['H'],
                    W=self.source_data_dict['W'],
                    K=self.source_data_dict['Ks'][image_source_index].clone(),
                    c2w=self.source_data_dict['poses'][image_source_index].clone(),
                    **render_kwargs_source
                )
                rays_d_source = rays_d_source / rays_d_source.norm(dim=-1, keepdim=True) # !!! important normalize
                # === Target ===
                image_target_index = pair['idx2']
                rays_o_target, rays_d_target = Model.get_rays(
                    H=self.target_data_dict['H'],
                    W=self.target_data_dict['W'],
                    K=self.target_data_dict['Ks'][image_target_index].clone(),
                    c2w=self.target_data_dict['poses'][image_target_index].clone(),
                    **render_kwargs_target
                )
                rays_d_target = rays_d_target / rays_d_target.norm(dim=-1, keepdim=True) # !!! important normalize
                # === Extract pixel coordinates from the filtered matches ===
                x_source = filtered_matches[:, 0, 0]
                y_source = filtered_matches[:, 0, 1]
                x_target = filtered_matches[:, 1, 0]
                y_target = filtered_matches[:, 1, 1]

                # === Ensure indices are torch tensors ===
                if not isinstance(x_source, torch.Tensor):
                    x_source = torch.from_numpy(x_source.astype(np.int64))
                    y_source = torch.from_numpy(y_source.astype(np.int64))
                    x_target = torch.from_numpy(x_target.astype(np.int64))
                    y_target = torch.from_numpy(y_target.astype(np.int64))

                # === Sample matched rays ===
                rays_o_source_matched = rays_o_source[y_source, x_source]  # shape (N, 3)
                rays_d_source_matched = rays_d_source[y_source, x_source]  # shape (N, 3)
                rays_o_target_matched = rays_o_target[y_target, x_target]  # shape (N, 3)
                rays_d_target_matched = rays_d_target[y_target, x_target]  # shape (N, 3)

                # === Append to the list ===
                rays_o_source_all.append(rays_o_source_matched)
                rays_d_source_all.append(rays_d_source_matched)
                rays_o_target_all.append(rays_o_target_matched)
                rays_d_target_all.append(rays_d_target_matched)

                # Store back into 'pair'
                pair["matches"] = filtered_matches
                pair["confidences"] = filtered_confidences

            # === Stack if not empty ===
            if len(rays_o_source_all) > 0:
                rays_o_source_all = torch.cat(rays_o_source_all, dim=0)  # (TotalN, 3)
                rays_d_source_all = torch.cat(rays_d_source_all, dim=0)  # (TotalN, 3)
                rays_o_target_all = torch.cat(rays_o_target_all, dim=0)  # (TotalN, 3)
                rays_d_target_all = torch.cat(rays_d_target_all, dim=0)  # (TotalN, 3)
            else:
                print("error! no pairs have been detected in this case")
                rays_o_source_all = torch.empty(0, 3)
                rays_d_source_all = torch.empty(0, 3)
                rays_o_target_all = torch.empty(0, 3)
                rays_d_target_all = torch.empty(0, 3)

            batch_size = 512
            # generate_surface_points_between_poses
            # Prepare lists to collect zero-SDF points from pose1 and pose2
            zero_sdf_source_list = []
            zero_sdf_target_list = []
            # import pdb; pdb.set_trace()

            # Number of total rays
            num_rays = rays_o_source_all.shape[0]  # same as rays_o_target_all.shape[0], presumably
            for start_idx in range(0, num_rays, batch_size):
                end_idx = start_idx + batch_size

                # === Source batch ===
                batch_rays_o_source = rays_o_source_all[start_idx:end_idx]
                batch_rays_d_source = rays_d_source_all[start_idx:end_idx]

                # === Target batch ===
                batch_rays_o_target = rays_o_target_all[start_idx:end_idx]
                batch_rays_d_target = rays_d_target_all[start_idx:end_idx]

                # === Query SDF zero points (target space)
                zero_sdf_points_target, target_valid_mask = self.generate_zero_sdf_points(
                    model=self.target_model,
                    rays_o=batch_rays_o_target,
                    rays_d=batch_rays_d_target,
                    cfg=self.target_net_cfg,
                    data_dict=self.target_data_dict,
                    zero_sdf_thereshold=1e-3,
                    inf_depth_thereshold=2.0,
                    return_all=True
                )

                # === Query SDF zero points (source space)
                zero_sdf_points_source, source_valid_mask = self.generate_zero_sdf_points(
                    model=self.source_model,
                    rays_o=batch_rays_o_source,
                    rays_d=batch_rays_d_source,
                    cfg = self.source_net_cfg,
                    data_dict = self.source_data_dict,
                    zero_sdf_thereshold=1e-3,
                    inf_depth_thereshold=2.0,
                    return_all=True
                )
                paired_valid_mask = source_valid_mask & target_valid_mask
                if torch.any(paired_valid_mask):
                    zero_sdf_source_list.append(zero_sdf_points_source[paired_valid_mask])
                    zero_sdf_target_list.append(zero_sdf_points_target[paired_valid_mask])

            # === Merge
            if len(zero_sdf_source_list) > 0:
                zero_sdf_source_all = torch.cat(zero_sdf_source_list, dim=0)
                zero_sdf_target_all = torch.cat(zero_sdf_target_list, dim=0)
            else:
                zero_sdf_source_all = torch.empty(0, 3, device=self.device)
                zero_sdf_target_all = torch.empty(0, 3, device=self.device)

            # === Export point clouds
            if save_debug:
                points_source = zero_sdf_source_all.detach().cpu().numpy()
                points_target = zero_sdf_target_all.detach().cpu().numpy()

                pcd_source = o3d.geometry.PointCloud()
                pcd_source.points = o3d.utility.Vector3dVector(points_source)

                pcd_target = o3d.geometry.PointCloud()
                pcd_target.points = o3d.utility.Vector3dVector(points_target)
                o3d.io.write_point_cloud(out_path_source, pcd_source)
                o3d.io.write_point_cloud(out_path_target, pcd_target)
                save_rays_as_ply(rays_o_source_all, rays_d_source_all, out_path_source +"rays_source.ply")
                save_rays_as_ply(rays_o_target_all, rays_d_target_all, out_path_target + "rays_target.ply")
            return zero_sdf_source_all, zero_sdf_target_all

    def export_coarse_debug(self, vis_folder=None):
        if vis_folder is None:
            return
        if not hasattr(self, "zeroset_points_source_pose") or self.zeroset_points_source_pose is None:
            return
        if not hasattr(self, "zeroset_points_target_pose") or self.zeroset_points_target_pose is None:
            return
        debug_path = Path(vis_folder)
        debug_path.mkdir(parents=True, exist_ok=True)

        coarse_rt = np.asarray(self.coarse_rt, dtype=np.float32)
        coarse_rt_torch = torch.from_numpy(coarse_rt).to(
            device=self.zeroset_points_source_pose.device,
            dtype=self.zeroset_points_source_pose.dtype,
        )
        source_pts = self.zeroset_points_source_pose
        target_pts = self.zeroset_points_target_pose
        source_pts_h = torch.cat(
            [source_pts, torch.ones((source_pts.shape[0], 1), device=source_pts.device, dtype=source_pts.dtype)],
            dim=1,
        )
        source_pts_coarse = (source_pts_h @ coarse_rt_torch.T)[:, :3]

        coarse_source_path = debug_path / f"{self.source_name}_coarse_rt.ply"
        target_ref_path = debug_path / f"{self.target_name}_reference_for_coarse.ply"
        utils.write_ply(source_pts_coarse.detach().cpu().numpy(), coarse_source_path.as_posix())
        utils.write_ply(target_pts.detach().cpu().numpy(), target_ref_path.as_posix())

        coarse_record = {}
        for key, value in self.coarse_best_result.items():
            if isinstance(value, np.ndarray):
                coarse_record[key] = value.tolist()
            elif isinstance(value, list):
                serialized_list = []
                for item in value:
                    if isinstance(item, dict):
                        serialized_item = {}
                        for sub_key, sub_val in item.items():
                            if isinstance(sub_val, np.ndarray):
                                serialized_item[sub_key] = sub_val.tolist()
                            else:
                                serialized_item[sub_key] = sub_val
                        serialized_list.append(serialized_item)
                    else:
                        serialized_list.append(item.tolist() if isinstance(item, np.ndarray) else item)
                coarse_record[key] = serialized_list
            else:
                coarse_record[key] = value
        coarse_record["source_name"] = self.source_name
        coarse_record["target_name"] = self.target_name
        coarse_record["coarse_source_ply"] = coarse_source_path.as_posix()
        coarse_record["target_reference_ply"] = target_ref_path.as_posix()
        with open(debug_path / "coarse_registration_record.json", "w", encoding="utf-8") as f:
            json.dump(coarse_record, f, indent=2, ensure_ascii=False)

    def _resolve_mesh_debug_path(self, exp_dir, fallback_mesh_path):
        exp_dir = Path(exp_dir)
        mesh_dir = exp_dir / "meshes"
        preferred_names = [
            "300000.ply",
            "00300000.ply",
        ]
        for preferred_name in preferred_names:
            candidate = mesh_dir / preferred_name
            if candidate.exists():
                return candidate
        final_mesh_candidates = sorted(mesh_dir.glob("*final_mesh.ply"))
        if final_mesh_candidates:
            return final_mesh_candidates[0]
        final_candidates = sorted(mesh_dir.glob("*final.ply"))
        if final_candidates:
            return final_candidates[0]
        if mesh_dir.exists():
            ply_candidates = sorted(mesh_dir.glob("*.ply"))
            if ply_candidates:
                return ply_candidates[0]
        fallback_path = Path(fallback_mesh_path)
        if fallback_path.exists():
            return fallback_path
        raise FileNotFoundError(f"Unable to resolve mesh debug path from {mesh_dir} or fallback {fallback_path}")

    def export_mesh_debug(self, vis_folder=None, fine_rt=None):
        if vis_folder is None:
            return
        debug_path = Path(vis_folder)
        debug_path.mkdir(parents=True, exist_ok=True)

        source_mesh_path = self._resolve_mesh_debug_path(
            self.config['source_exp_dir'],
            self.config['source_mesh_dir'],
        )
        target_mesh_path = self._resolve_mesh_debug_path(
            self.config['target_exp_dir'],
            self.config['target_mesh_dir'],
        )

        source_mesh = trimesh.load_mesh(source_mesh_path.as_posix())
        target_mesh = trimesh.load_mesh(target_mesh_path.as_posix())
        source_mesh_coarse = source_mesh.copy()
        source_mesh_coarse.apply_transform(np.asarray(self.coarse_rt, dtype=np.float64))

        target_reference_out = debug_path / f"{self.target_name}_mesh_reference.ply"
        source_raw_out = debug_path / f"{self.source_name}_mesh_raw.ply"
        source_coarse_out = debug_path / f"{self.source_name}_mesh_coarse_rt.ply"
        target_mesh.export(target_reference_out.as_posix())
        source_mesh.export(source_raw_out.as_posix())
        source_mesh_coarse.export(source_coarse_out.as_posix())

        fine_source_out = None
        if fine_rt is not None:
            source_mesh_fine = source_mesh.copy()
            source_mesh_fine.apply_transform(np.asarray(fine_rt, dtype=np.float64))
            fine_source_out = debug_path / f"{self.source_name}_mesh_fine_rt.ply"
            source_mesh_fine.export(fine_source_out.as_posix())

        payload = {
            "source_mesh_input": source_mesh_path.as_posix(),
            "target_mesh_input": target_mesh_path.as_posix(),
            "source_mesh_raw": source_raw_out.as_posix(),
            "source_mesh_coarse_rt": source_coarse_out.as_posix(),
            "target_mesh_reference": target_reference_out.as_posix(),
            "coarse_rt": np.asarray(self.coarse_rt, dtype=np.float32).tolist(),
        }
        if fine_rt is not None and fine_source_out is not None:
            payload["source_mesh_fine_rt"] = fine_source_out.as_posix()
            payload["fine_rt"] = np.asarray(fine_rt, dtype=np.float32).tolist()
        with open(debug_path / "mesh_debug_record.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def Fine_Registration(self):
        print("===================== Start Fine Registration ==================================")
        self.optimizer = torch.optim.Adam(
            [
                {'params': getattr(self, 'raw_translation'), 'lr': 1e-4},
                {'params': getattr(self, 'raw_quaternion'), 'lr': 1e-4},
            ],
            amsgrad=False
        )
        train_iters = 1000  # need to add to cfg later
        pbar = trange(0, train_iters)
        for _ in pbar:
            self.optimizer.zero_grad(set_to_none=True)
            # use sdf loss for reg, no need to visit images
            # 将 target 点云从 target 空间转换到 source 空间 (即 delta_RT^{-1} 应用)
            zero_sdf_source_in_target_space = transfer_points_to_local_axis(
                self.zeroset_points_target_pose,
                translation=self.raw_translation,
                quaternion=self.raw_quaternion
            )
            # source zeroset points 在 target sdf 中的值
            sdf_source_in_target = self.target_model.sdf(
                zero_sdf_source_in_target_space).contiguous().squeeze()
            sdf_source_in_target_abs = torch.abs(sdf_source_in_target.view(-1, 1))  # shape (N,1)
            robust_loss = self.adaptive_loss.lossfun(sdf_source_in_target_abs.mean().view(-1, 1))
            R_ek_loss = torch.abs(torch.norm(self.raw_quaternion) - 1)
            global_loss = robust_loss + R_ek_loss
            global_loss.backward()
            self.optimizer.step()
            print('raw_translation: {}, raw_quaternion: {}, loss: {}'.format(self.raw_translation,self.raw_quaternion, loss.norm()))
            print('trans gradient : {}, quad gradient : {}, loss: {}'.format(self.raw_translation.grad,self.raw_quaternion.grad,loss.norm()))


    def Fine_Registration(self):
        print("===================== Start Fine Registration ==================================")
        self.optimizer = torch.optim.Adam(
            [
                {'params': getattr(self, 'raw_translation'), 'lr': 1e-4},
                {'params': getattr(self, 'raw_quaternion'), 'lr': 1e-4},
                {'params': list(getattr(self, 'adaptive_loss').parameters()), 'lr': 1e-2},
            ],
            amsgrad=False
        )
        train_iters = 5000
        pbar = trange(0, train_iters)
        for _ in pbar:
            self.optimizer.zero_grad(set_to_none=True)
            # Optimize the source->target rigid transform with bi-directional SDF constraints.
            source_in_target = transfer_points_to_local_axis(
                self.zeroset_points_source_pose,
                translation=self.raw_translation,
                quaternion=self.raw_quaternion
            )

            transform = torch.eye(4, device=self.device, dtype=self.raw_translation.dtype)
            transform[:3, :3] = quaternion_to_rotation_matrix_torch(
                self.raw_quaternion, device=self.raw_quaternion.device
            )
            transform[:3, 3] = self.raw_translation
            transform_inv = torch.inverse(transform)

            target_pts_h = torch.cat(
                [
                    self.zeroset_points_target_pose,
                    torch.ones(
                        (self.zeroset_points_target_pose.shape[0], 1),
                        device=self.zeroset_points_target_pose.device,
                        dtype=self.zeroset_points_target_pose.dtype,
                    ),
                ],
                dim=1,
            )
            target_in_source = (target_pts_h @ transform_inv.T)[:, :3]

            sdf_source_in_target = self.target_model.sdf(source_in_target).contiguous().reshape(-1, 1)
            sdf_target_in_source = self.source_model.sdf(target_in_source).contiguous().reshape(-1, 1)

            sdf_source_in_target_abs = torch.abs(sdf_source_in_target)
            sdf_target_in_source_abs = torch.abs(sdf_target_in_source)
            sdf_abs_mean_st = sdf_source_in_target_abs.mean()
            sdf_abs_mean_ts = sdf_target_in_source_abs.mean()
            sdf_abs_mean = 0.5 * (sdf_abs_mean_st + sdf_abs_mean_ts)
            quaternion_norm_loss = torch.abs(torch.norm(self.raw_quaternion) - 1)
            robust_input = torch.stack([sdf_abs_mean_st, sdf_abs_mean_ts], dim=0).view(-1, 1)
            robust_loss = self.adaptive_loss.lossfun(robust_input).mean()
            global_loss = robust_loss + quaternion_norm_loss
            global_loss.backward()
            self.optimizer.step()
            pbar.set_postfix(
                loss=float(global_loss.detach().item()),
                robust=float(robust_loss.detach().item()),
                sdf_abs=float(sdf_abs_mean.detach().item()),
                sdf_st=float(sdf_abs_mean_st.detach().item()),
                sdf_ts=float(sdf_abs_mean_ts.detach().item()),
                qnorm=float(quaternion_norm_loss.detach().item())
            )

        final_rt = np.eye(4, dtype=np.float32)
        q_final = self.raw_quaternion.detach()
        q_final = q_final / torch.clamp(torch.norm(q_final), min=1e-12)
        final_rt[:3, :3] = quaternion_to_rotation_matrix_torch(
            q_final, device=q_final.device
        ).cpu().numpy()
        final_rt[:3, 3] = self.raw_translation.detach().cpu().numpy()
        return final_rt

    @torch.no_grad()
    def render_single_image(self, model, width, height, K, pose, cfg, data_dict, savedir):
        render_kwargs = {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': cfg.fine_model_and_render.stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_grad': True,
                'render_depth': True,
                'render_in_out': True,
            }
        render_kwargs.update(dict(
            savedir=savedir,
            eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False,
            rgb_only=True,
        ))
        rays_o, rays_d, viewdirs = Model.get_rays_of_a_view(
            height, width, K, pose, cfg.data.ndc, inverse_y=render_kwargs['inverse_y'],
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'disp', 'alphainv_cum']
        rays_o = rays_o.flatten(0, -2)
        rays_d = rays_d.flatten(0, -2)
        viewdirs = viewdirs.flatten(0, -2)
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(height, width,-1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        if savedir is not None:
            imageio.imwrite(os.path.join(savedir, 'test.png'), utils.to8b(rgb))
            print(">>>>> Image is saved to {}".format(os.path.join(savedir, 'test.png')))
        return rgb # this is a cpu numpy

    # @torch.no_grad()
    # def generate_zero_level_points_np(self, model):
    # TODO: replace this sdf query into the AABB based
    def sdf(self, model, pts, smooth_sdf=False, require_grad = False):
        # import pdb; pdb.set_trace()
        sdf_grid = model.smooth_conv(model.sdf.grid) if smooth_sdf else model.sdf.grid
        sample_points = pts
        if require_grad:
            sdf, gradient, _ = model.grid_sampler(sample_points, sdf_grid, sample_ret=True, sample_grad=require_grad, displace=1.0)
            return sdf.detach().clone(), gradient.detach().clone()
            # return sdf.detach().clone().cpu().numpy(), gradient.detach().clone().cpu().numpy()
        else:
            sdf = model.grid_sampler(sample_points, sdf_grid)
            return sdf.detach().clone()
            # return sdf.detach().clone().cpu().numpy()

    def get_sdf_grid(self, model, smooth_sdf=False, save_sdf_grid=True):
        sdf_grid = model.smooth_conv(model.sdf.grid) if smooth_sdf else model.sdf.grid
        sdf_grid = sdf_grid.clone().squeeze()
        if save_sdf_grid:
            file_path = "./test_sdf_grid.npy"
            sdf_np = sdf_grid.detach().cpu().numpy()
            np.save(file_path, sdf_np)
            print("saved at ", file_path)
        return sdf_grid

    def gen_rays_at_pose_mat(self, transform_matrix, resolution_level=1, intrinsic_inv=None, H=None, W=None,
                             is_np=False):
        if is_np:
            transform_matrix = torch.from_numpy(transform_matrix.astype(np.float32)).to(self.device)

        if intrinsic_inv is None:
            intrinsic_inv = self.intrinsics_all_inv[0]  # fallback

        l = resolution_level
        tx = torch.linspace(0, W - 1, W // l, device=transform_matrix.device)
        ty = torch.linspace(0, H - 1, H // l, device=transform_matrix.device)
        pixels_x, pixels_y = torch.meshgrid(tx, ty, indexing='ij')

        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(intrinsic_inv[None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(transform_matrix[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = transform_matrix[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)  # H W 3

    def generate_zero_sdf_points(self, model, rays_o, rays_d, cfg, data_dict, zero_sdf_thereshold=1e-3, inf_depth_thereshold=4.0,
                                 return_all=True):
        """
        Compute surface intersection points by progressive ray marching given rays and SDF field.

        Args:
            model: The model to query sdf, should be self.source_model or self.target_model.
            rays_o: (N,3) tensor, ray origins.
            rays_d: (N,3) tensor, ray directions.
            zero_sdf_thereshold: Convergence threshold for zero-sdf.
            inf_depth_thereshold: Early termination threshold for diverged rays.
            return_all: Whether to return all sampled points or only those satisfying the sdf < threshold.

        Returns:
            zero_points: (M,3) tensor, the computed surface points.
            zero_mask: (N,) bool tensor, indicating valid converged points.
        """
        render_kwargs = {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': cfg.fine_model_and_render.stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_grad': True,
                'render_depth': True,
                'render_in_out': True,
        }
        render_kwargs.update(dict(
            savedir='savedir',
            eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False,
            rgb_only=True,
        ))
        # this is a special adjustment to align rays o to the aabb of within the grid to avoid mis-sdf value
        ray_pts, ray_id, step_id, mask_outbbox, N_steps = model.sample_ray( rays_o = rays_o, rays_d=rays_d, **render_kwargs)
        # Step 1: 过滤出落在 bbox 内的采样点
        # valid_mask = ~mask_outbbox # have been updated
        ray_pts = ray_pts.detach()
        ray_id = ray_id.detach()
        ray_dict = defaultdict(list)
        for pt, rid in zip(ray_pts, ray_id):
            ray_dict[rid.item()].append(pt)

        # Step 2: For each ray, find the point with minimal SDF
        first_valid_pts = {}
        rays_o_indices = []

        for rid, pts in ray_dict.items():
            pts_tensor = torch.stack(pts, dim=0).to(ray_pts.device)  # [L, 3]
            sdf_vals = model.sdf(pts_tensor).squeeze(-1)  # [L]

            # 筛选 SDF > 0 的点
            positive_mask = sdf_vals > 0
            sdf_vals_filtered = sdf_vals[positive_mask]
            pts_filtered = pts_tensor[positive_mask]

            # 获取最小 sdf > 0 的点（你假设至少有一个）
            min_idx = torch.argmin(sdf_vals_filtered)
            best_pt = pts_filtered[min_idx]

            first_valid_pts[rid] = best_pt
            rays_o_indices.append(rid)
        # Step 3: Pack output
        print("pre filtered the points ")
        rays_o_trans_to_sdfmin = torch.stack([first_valid_pts[rid] for rid in rays_o_indices], dim=0)  # [N_valid, 3]
        rays_d_filtered = rays_d[rays_o_indices]  # [N_valid, 3]
        rays_o = rays_o_trans_to_sdfmin
        rays_d = rays_d_filtered
        # filter first, then generate
        device = rays_o.device
        depths = torch.zeros(len(rays_o), dtype=torch.float32, device=device)
        sdfs = torch.ones(len(rays_o), dtype=torch.float32, device=device)
        rays_o = rays_o.clone()  # Very important to keep the original
        max_iters = 30
        step = 0
        rays_mask = torch.ones_like(sdfs, dtype=torch.bool)

        while torch.sum(rays_mask) > 0 and step < max_iters:
            pts, dirs = rays_o[rays_mask], rays_d[rays_mask]

            tmp_sdfs = self.sdf(model, pts, smooth_sdf=False, require_grad=False).squeeze()  # [N_active]
            pts = pts + dirs * tmp_sdfs.unsqueeze(-1)  # 沿当前 sdf 步进
            rays_o[rays_mask] = pts
            depths[rays_mask] = depths[rays_mask] + tmp_sdfs
            sdfs[rays_mask] = tmp_sdfs

            # 判断是否已经收敛（在 zero 面附近）或发散
            zero_mask = (sdfs > -zero_sdf_thereshold) & (sdfs < zero_sdf_thereshold)
            inf_mask = (sdfs.abs() > inf_depth_thereshold)

            rays_mask = ~(zero_mask | inf_mask)  # 继续 marching 的 ray
            # step += 1
            print(step)

        # 超出最大迭代次数还未收敛的，强制标记为发散
        if return_all:
            zero_points = rays_o
        elif torch.sum(zero_mask) > 0:
            zero_points = rays_o[zero_mask]
        else:
            print("WARNING: No points are generated from SDF field")
            zero_points = None

        return zero_points, zero_mask

    def _points_in_bbox(self, pts, xyz_min, xyz_max, eps=1e-6):
        return ((pts >= (xyz_min - eps)) & (pts <= (xyz_max + eps))).all(dim=-1)

    @torch.no_grad()
    def _refine_zero_crossing(self, model, p0, p1, s0, s1, iters=8):
        left_pt, right_pt = p0.clone(), p1.clone()
        left_sdf = s0.clone()
        right_sdf = s1.clone()

        for _ in range(iters):
            mid_pt = 0.5 * (left_pt + right_pt)
            if not self._points_in_bbox(mid_pt.unsqueeze(0), model.xyz_min, model.xyz_max)[0]:
                break

            mid_sdf = self.sdf(model, mid_pt.unsqueeze(0), smooth_sdf=False, require_grad=False).reshape(-1)[0]
            if torch.abs(mid_sdf) <= 1e-6:
                return mid_pt

            if left_sdf * mid_sdf <= 0:
                right_pt = mid_pt
                right_sdf = mid_sdf
            else:
                left_pt = mid_pt
                left_sdf = mid_sdf

        denom = left_sdf - right_sdf
        if torch.abs(denom) < 1e-8:
            return 0.5 * (left_pt + right_pt)

        ratio = torch.clamp(left_sdf / denom, 0.0, 1.0)
        return left_pt + ratio * (right_pt - left_pt)

    def generate_zero_sdf_points(self, model, rays_o, rays_d, cfg, data_dict, zero_sdf_thereshold=1e-3, inf_depth_thereshold=4.0,
                                 return_all=True):
        """
        Compute zero-level points by detecting sign changes along sampled in-bbox ray points.
        """
        render_kwargs = {
                'near': data_dict['near'],
                'far': data_dict['far'],
                'bg': 1 if cfg.data.white_bkgd else 0,
                'stepsize': cfg.fine_model_and_render.stepsize,
                'inverse_y': cfg.data.inverse_y,
                'flip_x': cfg.data.flip_x,
                'flip_y': cfg.data.flip_y,
                'render_grad': True,
                'render_depth': True,
                'render_in_out': True,
        }
        render_kwargs.update(dict(
            savedir='savedir',
            eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False,
            rgb_only=True,
        ))

        num_rays = rays_o.shape[0]
        all_zero_points = torch.zeros((num_rays, 3), dtype=rays_o.dtype, device=rays_o.device)
        valid_mask = torch.zeros(num_rays, dtype=torch.bool, device=rays_o.device)

        # Outside the grid AABB, grid_sample pads with zeros. Those padded zeros are not real surface hits.
        ray_pts, ray_id, step_id, mask_outbbox, N_steps = model.sample_ray(rays_o=rays_o, rays_d=rays_d, **render_kwargs)
        if ray_pts.numel() == 0:
            if return_all:
                return all_zero_points, valid_mask
            return all_zero_points[valid_mask], valid_mask

        ray_pts = ray_pts.detach()
        ray_id = ray_id.detach()
        sdf_vals = model.sdf(ray_pts).reshape(-1).detach()

        for rid in torch.unique_consecutive(ray_id).tolist():
            ray_mask = ray_id == rid
            pts = ray_pts[ray_mask]
            sdfs = sdf_vals[ray_mask]
            if pts.shape[0] == 0:
                continue

            min_abs_sdf, min_abs_idx = torch.min(sdfs.abs(), dim=0)
            if min_abs_sdf <= zero_sdf_thereshold:
                all_zero_points[rid] = pts[min_abs_idx]
                valid_mask[rid] = True
                continue

            sign_change_idx = torch.where((sdfs[:-1] * sdfs[1:]) <= 0)[0]
            if sign_change_idx.numel() == 0:
                continue

            first_idx = sign_change_idx[0]
            p0, p1 = pts[first_idx], pts[first_idx + 1]
            s0, s1 = sdfs[first_idx], sdfs[first_idx + 1]
            zero_pt = self._refine_zero_crossing(model, p0, p1, s0, s1)

            if not self._points_in_bbox(zero_pt.unsqueeze(0), model.xyz_min, model.xyz_max)[0]:
                continue

            sdf_at_zero = self.sdf(model, zero_pt.unsqueeze(0), smooth_sdf=False, require_grad=False).reshape(-1)[0]
            if torch.abs(sdf_at_zero) > inf_depth_thereshold:
                continue

            all_zero_points[rid] = zero_pt
            valid_mask[rid] = True

        if return_all:
            return all_zero_points, valid_mask

        zero_points = all_zero_points[valid_mask]
        if zero_points.shape[0] == 0:
            print("WARNING: No points are generated from SDF field")
        return zero_points, valid_mask


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    if args.config is None:
        print("Please specify the config file")
        sys.exit(1)
    os.chdir(ROOT)
    config = load_registration_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError('Voxurf registration requires a CUDA GPU')
    print("load config from {}".format(args.config))

    device = None
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda:0')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        device = torch.device

    mp_reg = MPReg(config, device)

"""
python mp_reg.py --config ./configs/reg/dadian12.json
"""

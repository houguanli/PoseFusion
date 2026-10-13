#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
from pathlib import Path

import cv2 as cv
import numpy as np
from project_paths import ROOT, load_registration_config


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


def clean_case_name(case_name):
    return re.sub(r"_(?:pose|pos|position|p)\d*$", "", case_name)


def get_images_and_masks(case_dir):
    image_dir = Path(case_dir) / "image"
    mask_dir = Path(case_dir) / "mask"
    if not image_dir.is_dir():
        raise ValueError(f"Image directory not found: {image_dir}")
    if not mask_dir.is_dir():
        raise ValueError(f"Mask directory not found: {mask_dir}")

    images = sorted([p.name for p in image_dir.glob("*.png")])
    masks = sorted([p.name for p in mask_dir.glob("*.png")])
    if len(images) == 0 or len(masks) == 0:
        raise ValueError(f"Missing png files in {case_dir}")
    if len(images) != len(masks):
        raise ValueError(f"Image count {len(images)} != mask count {len(masks)} in {case_dir}")
    return images, masks


def load_npz_cameras(npz_path, n_images):
    camera_dict = np.load(npz_path)
    world_mats = [camera_dict[f"world_mat_{idx}"].astype(np.float32) for idx in range(n_images)]
    scale_mats = [camera_dict[f"scale_mat_{idx}"].astype(np.float32) for idx in range(n_images)]
    return camera_dict, world_mats, scale_mats


def get_poses_from_npz(npz_path):
    data = np.load(npz_path)
    poses = []
    keys = sorted((k for k in data.files if k.startswith('world_mat_') and not k.startswith('world_mat_inv_')),
                  key=lambda k: int(k.rsplit('_', 1)[1]))
    for key in keys:
        idx = int(key.rsplit('_', 1)[1])
        _, pose = load_K_Rt_from_P(None, (data[key] @ data[f'scale_mat_{idx}'])[:3, :4])
        poses.append(pose)
    return poses


def compute_normalized_camera_distance(target_poses):
    if len(target_poses) <= 1:
        return 1.0
    distances = []
    for i in range(len(target_poses)):
        for j in range(i + 1, len(target_poses)):
            pos_i = target_poses[i][:3, 3]
            pos_j = target_poses[j][:3, 3]
            distances.append(np.linalg.norm(pos_i - pos_j))
    return float(np.mean(distances)) if distances else 1.0


def measure_camera_similarity_normalized(transformed_source_pose, target_pose, normalize_distance):
    optical_axis_source = transformed_source_pose[:3, 2]
    optical_axis_target = target_pose[:3, 2]
    optical_axis_source = optical_axis_source / np.linalg.norm(optical_axis_source)
    optical_axis_target = optical_axis_target / np.linalg.norm(optical_axis_target)
    cos_angle = np.clip(np.dot(optical_axis_source, optical_axis_target), -1.0, 1.0)
    viewing_angle_rad = np.arccos(cos_angle)

    M_rel = np.linalg.inv(transformed_source_pose) @ target_pose
    translation_diff = np.linalg.norm(M_rel[:3, 3])
    translation_normalized = translation_diff / normalize_distance
    similarity = viewing_angle_rad + translation_normalized
    return float(similarity), float(viewing_angle_rad), float(translation_normalized)


def filter_source_images_by_camera_overlap(target_poses, source_poses, delta_RT, similarity_threshold=0.5):
    normalize_distance = compute_normalized_camera_distance(target_poses)
    keep_indices = []
    discard_indices = []
    details = []

    for i, source_pose in enumerate(source_poses):
        transformed_pose = delta_RT @ source_pose
        min_similarity = float("inf")
        closest_target_idx = -1
        min_viewing_angle = 0.0
        min_translation_norm = 0.0
        for j, target_pose in enumerate(target_poses):
            similarity, viewing_angle_rad, translation_norm = measure_camera_similarity_normalized(
                transformed_pose, target_pose, normalize_distance
            )
            if similarity < min_similarity:
                min_similarity = similarity
                closest_target_idx = j
                min_viewing_angle = viewing_angle_rad
                min_translation_norm = translation_norm

        detail = {
            "source_index": i,
            "closest_target_index": closest_target_idx,
            "similarity": float(min_similarity),
            "viewing_angle_deg": float(np.degrees(min_viewing_angle)),
            "translation_normalized": float(min_translation_norm),
        }
        if min_similarity > similarity_threshold:
            keep_indices.append(i)
            detail["decision"] = "keep"
        else:
            discard_indices.append(i)
            detail["decision"] = "discard"
        details.append(detail)

    return keep_indices, discard_indices, details


def load_config(path):
    return load_registration_config(path)


def get_case_name_from_datadir(datadir):
    p = Path(datadir)
    return f'{p.parent.name}_{p.name}' if re.fullmatch(r'pose[123]', p.name) else p.name


def get_transform_from_config(cfg):
    if "fine_rt" in cfg:
        return np.asarray(cfg["fine_rt"], dtype=np.float32)
    if "fine_translation" in cfg and "fine_quaternion" in cfg:
        raise ValueError("fine_translation/fine_quaternion path is not implemented; use fine_rt")
    if "raw_reg" in cfg:
        return np.asarray(cfg["raw_reg"], dtype=np.float32)
    raise ValueError("Config must contain fine_rt or raw_reg")


def fusion_dataset(config_files, output_name=None, similarity_threshold=0.5):
    configs = [load_config(p) for p in config_files]
    if not configs:
        raise ValueError("No configs provided")

    target_case = get_case_name_from_datadir(configs[0]["target_datadir"])
    for cfg in configs[1:]:
        if get_case_name_from_datadir(cfg["target_datadir"]) != target_case:
            raise ValueError("All configs must share the same target case")

    target_dir = Path(configs[0]["target_datadir"]).resolve()
    target_images, target_masks = get_images_and_masks(target_dir)
    _, target_world_mats, target_scale_mats = load_npz_cameras(target_dir / "cameras_sphere.npz", len(target_images))
    target_poses = get_poses_from_npz(target_dir / "cameras_sphere.npz")
    target_ref_img = cv.imread(str(target_dir / "image" / target_images[0]))
    target_h, target_w = target_ref_img.shape[:2]

    if output_name is None:
        output_name = f"{clean_case_name(target_case)}_full_est_s"
    if Path(output_name).name != output_name or output_name in ('.', '..'):
        raise ValueError('output_name must be a single folder name')
    output_dir = ROOT / 'outputs' / 'fusion' / output_name
    # Validate transforms before creating any output folders.
    for cfg in configs:
        get_transform_from_config(cfg)
    if output_dir.exists():
        raise FileExistsError(f'Refusing to replace existing fusion: {output_dir}')
    (output_dir / "image").mkdir(parents=True, exist_ok=True)
    (output_dir / "mask").mkdir(parents=True, exist_ok=True)

    all_world_mats = {}
    all_scale_mats = {}
    metadata = []
    image_index = 0

    for i, (img_file, mask_file) in enumerate(zip(target_images, target_masks)):
        img = cv.imread(str(target_dir / "image" / img_file))
        mask = cv.imread(str(target_dir / "mask" / mask_file))
        cv.imwrite(str(output_dir / "image" / f"{image_index:03d}.png"), img)
        cv.imwrite(str(output_dir / "mask" / f"{image_index:03d}.png"), mask)
        all_world_mats[image_index] = target_world_mats[i] @ target_scale_mats[i]
        all_scale_mats[image_index] = np.eye(4, dtype=np.float32)
        metadata.append({
            "fused_index": image_index,
            "case_name": target_case,
            "original_index": i,
            "role": "target",
        })
        image_index += 1

    source_summaries = []
    for cfg in configs:
        source_case = get_case_name_from_datadir(cfg["source_datadir"])
        source_dir = Path(cfg["source_datadir"]).resolve()
        source_images, source_masks = get_images_and_masks(source_dir)
        _, source_world_mats, source_scale_mats = load_npz_cameras(source_dir / "cameras_sphere.npz", len(source_images))
        source_poses = get_poses_from_npz(source_dir / "cameras_sphere.npz")
        RT = get_transform_from_config(cfg)
        keep_indices, discard_indices, filter_details = filter_source_images_by_camera_overlap(
            target_poses, source_poses, RT, similarity_threshold=similarity_threshold
        )

        source_summaries.append({
            "case_name": source_case,
            "keep_count": len(keep_indices),
            "discard_count": len(discard_indices),
            "filter_details": filter_details,
        })

        RT_inv = np.linalg.inv(RT)
        for i, (img_file, mask_file) in enumerate(zip(source_images, source_masks)):
            if i not in keep_indices:
                continue
            img = cv.imread(str(source_dir / "image" / img_file))
            mask = cv.imread(str(source_dir / "mask" / mask_file))
            src_h, src_w = img.shape[:2]
            sx = target_w / float(src_w)
            sy = target_h / float(src_h)
            if (src_h, src_w) != (target_h, target_w):
                img = cv.resize(img, (target_w, target_h), interpolation=cv.INTER_LINEAR)
                mask = cv.resize(mask, (target_w, target_h), interpolation=cv.INTER_NEAREST)
            cv.imwrite(str(output_dir / "image" / f"{image_index:03d}.png"), img)
            cv.imwrite(str(output_dir / "mask" / f"{image_index:03d}.png"), mask)

            world_mat = source_world_mats[i]
            scale_mat = source_scale_mats[i]
            new_world_mat = world_mat @ scale_mat @ RT_inv
            if (src_h, src_w) != (target_h, target_w):
                new_world_mat = new_world_mat.copy()
                new_world_mat[0, :] *= sx
                new_world_mat[1, :] *= sy
            all_world_mats[image_index] = new_world_mat
            all_scale_mats[image_index] = np.eye(4, dtype=np.float32)
            metadata.append({
                "fused_index": image_index,
                "case_name": source_case,
                "original_index": i,
                "role": "source",
                "similarity_threshold": similarity_threshold,
                "source_resolution": [int(src_h), int(src_w)],
                "fused_resolution": [int(target_h), int(target_w)],
            })
            image_index += 1

    npz_data = {}
    for idx in sorted(all_world_mats.keys()):
        npz_data[f"world_mat_{idx}"] = all_world_mats[idx]
        npz_data[f"scale_mat_{idx}"] = all_scale_mats[idx]
    np.savez(output_dir / "cameras_sphere.npz", **npz_data)

    summary = {
        "output_name": output_name,
        "target_case": target_case,
        "target_count": len(target_images),
        "total_count": len(metadata),
        "similarity_threshold": similarity_threshold,
        "config_files": [str(Path(p).resolve()) for p in config_files],
        "source_summaries": source_summaries,
    }
    with open(output_dir / "fusion_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(output_dir / "fusion_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return output_dir, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--output_name", default=None)
    parser.add_argument("--similarity_threshold", type=float, default=0.5)
    args = parser.parse_args()

    output_dir, summary = fusion_dataset(
        config_files=args.configs,
        output_name=args.output_name,
        similarity_threshold=args.similarity_threshold,
    )
    print(f"Output dataset: {output_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

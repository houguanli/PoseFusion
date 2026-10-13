import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import imageio

# This function is borrowed from IDR: https://github.com/lioryariv/idr
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

def load_dtu_data(basedir, normalize=True, reso_level=2, mask=True, white_bg=True):
    """Load indexed NeuS images with per-frame intrinsics and OpenCV c2w.

    mask=False retains actual RGB backgrounds while returning foreground masks.
    Resize each frame before stacking to bound memory; masks use nearest sampling.
    """
    from pathlib import Path
    root = Path(basedir)
    rgb_paths = sorted((root / 'image').glob('*.png'))
    if not rgb_paths:
        rgb_paths = sorted((root / 'image').glob('*.jpg'))
    if not rgb_paths:
        rgb_paths = sorted((root / 'rgb').glob('*.png'))
    mask_paths = sorted((root / 'mask').glob('*.png'))
    if not mask_paths:
        mask_paths = sorted((root / 'mask').glob('*.jpg'))
    if not rgb_paths or len(rgb_paths) != len(mask_paths):
        raise ValueError(f'Missing or mismatched image/mask pairs: {root}')
    if [p.stem for p in rgb_paths] != [p.stem for p in mask_paths]:
        raise ValueError(f'Image/mask filenames differ: {root}')
    if [int(p.stem) for p in rgb_paths] != list(range(len(rgb_paths))):
        raise ValueError(f'Frames must have consecutive indices starting at zero: {root}')
    if reso_level < 1:
        raise ValueError('reso_level must be >= 1')
    camera_name = 'cameras_sphere.npz' if normalize else 'cameras_large.npz'
    all_imgs, all_masks, all_poses, all_intrinsics = [], [], [], []
    original_shape, first_scale = None, None
    with np.load(root / camera_name) as camera:
        ids = sorted(int(k.rsplit('_', 1)[1]) for k in camera.files
                     if k.startswith('world_mat_') and not k.startswith('world_mat_inv_'))
        if ids != list(range(len(rgb_paths))):
            raise ValueError(f'Camera/frame indices differ: {root}')
        for i, (image_path, mask_path) in enumerate(zip(rgb_paths, mask_paths)):
            image = imageio.v2.imread(image_path)[..., :3]
            silhouette = imageio.v2.imread(mask_path)
            if silhouette.ndim == 3:
                silhouette = silhouette[..., :3].max(axis=-1)
            h0, w0 = image.shape[:2]
            if silhouette.shape != (h0, w0):
                raise ValueError(f'Mask size differs from image: {mask_path}')
            if original_shape is None:
                original_shape = (h0, w0)
            if original_shape != (h0, w0):
                raise ValueError(f'Inconsistent image dimensions within pose: {root}')
            scale = camera[f'scale_mat_{i}'].astype(np.float64) if normalize else np.eye(4)
            if first_scale is None:
                first_scale = scale.copy()
            if not np.allclose(scale, first_scale, atol=1e-6):
                raise ValueError(f'Scale matrices must be constant within a pose: {root}')
            projection = (camera[f'world_mat_{i}'].astype(np.float64) @ scale)[:3, :4]
            intrinsic, pose = load_K_Rt_from_P(None, projection)
            h, w = max(1, int(h0 / reso_level)), max(1, int(w0 / reso_level))
            image = image.astype(np.float32) / 255.
            silhouette = silhouette.astype(np.float32) / 255.
            if mask:
                bg = 1. if white_bg else 0.
                image = image * silhouette[..., None] + bg * (1 - silhouette[..., None])
            if (h, w) != (h0, w0):
                image = cv.resize(image, (w, h), interpolation=cv.INTER_AREA)
                silhouette = cv.resize(silhouette, (w, h), interpolation=cv.INTER_NEAREST)
            intrinsic[0, :] *= w / w0
            intrinsic[1, :] *= h / h0
            all_imgs.append(image)
            all_masks.append(silhouette[..., None])
            all_poses.append(pose)
            all_intrinsics.append(intrinsic)
    imgs = np.stack(all_imgs).astype(np.float32)
    masks = np.stack(all_masks).astype(np.float32)
    poses = np.stack(all_poses).astype(np.float32)
    intrinsics = np.stack(all_intrinsics).astype(np.float32)
    held_out = [8,13,16,21,26,31,34] + ([56] if len(imgs) >= 80 else [])
    i_test = np.array([i for i in held_out if i < len(imgs)], dtype=np.int64)
    i_train = np.setdiff1d(np.arange(len(imgs)), i_test)
    i_split = [i_train, i_test.copy(), i_test]
    return imgs, poses, poses[i_test], [h,w,float(intrinsics[0,0,0])], intrinsics, i_split, first_scale.astype(np.float32), masks

class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        self.n_images = len(self.images_lis)
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, 'mask/*.png')))
        self.masks_np = np.stack([cv.imread(im_name) for im_name in self.masks_lis]) / 256.0

        # world_mat is a projection matrix from world to image
        self.world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cpu()   # [n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))['scale_mat_0']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import trimesh


def quat_to_matrix(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])

class Rect: # _right_hand
    def __init__(self, with_init=False, o = None, x = None, y = None, z = None):
        if not with_init:
            self.o, self.x, self.y, self.z = np.array(3), np.array(3), np.array(3), np.array(3)
        else:
            self.o, self.x, self.y, self.z = o, x, y, z

class Mesh2SDF:
    def __init__(self, mesh_path=None, mesh=None):
        if mesh is not None:
            self.mesh = mesh
        elif mesh_path is not None:
            self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        else:
            raise ValueError("Either mesh_path or mesh must be provided.")

        self.aabb_tree = o3d.geometry.KDTreeFlann(self.mesh)
        self.obb = None

    def calculate_sdf(self, points):
        sdf_values = []
        for point in points:
            closest_point_id = self.aabb_tree.search_knn_vector_3d(point, 1)[1][0]
            closest_point = np.asarray(self.mesh.vertices)[closest_point_id]
            sdf_value = np.linalg.norm(closest_point - point)
            sdf_values.append(sdf_value)
        return np.array(sdf_values)

    def calculate_obb_from_point_cloud(self):
        pointArray = np.array(self.mesh.vertices)
        ca = np.cov(pointArray, y=None, rowvar=0, bias=1)
        v, vect = np.linalg.eig(ca)
        tvect = np.transpose(vect)
        # use the inverse of the eigenvectors as a rotation matrix and
        # rotate the points so they align with the x and y axes
        ar = np.dot(pointArray, np.linalg.inv(tvect))

        # get the minimum and maximum x and y
        mina = np.min(ar, axis=0)
        maxa = np.max(ar, axis=0)
        diff = (maxa - mina) * 0.5
        # the center is just half way between the min and max xy
        center = mina + diff

        # get the 8 corners by subtracting and adding half the bounding boxes height and width to the center
        pointShape = pointArray.shape
        if pointShape[1] == 2:
            corners = np.array([center + [-diff[0], -diff[1]],
                                center + [diff[0], -diff[1]],
                                center + [diff[0], diff[1]],
                                center + [-diff[0], diff[1]],
                                center + [-diff[0], -diff[1]]])
        if pointShape[1] == 3:
            # get the 8 corners by subtracting and adding half the bounding boxes height and width to the center
            corners = np.array([center + [-diff[0], -diff[1], -diff[2]],
                                center + [diff[0], -diff[1], -diff[2]],
                                center + [diff[0], diff[1], -diff[2]],
                                center + [-diff[0], diff[1], -diff[2]],
                                center + [-diff[0], diff[1], diff[2]],
                                center + [diff[0], diff[1], diff[2]],
                                center + [diff[0], -diff[1], diff[2]],
                                center + [-diff[0], -diff[1], diff[2]],
                                center + [-diff[0], -diff[1], -diff[2]]])

            # use the the eigenvectors as a rotation matrix and
        # rotate the corners and the centerback
        corners = np.dot(corners, tvect)
        center = np.dot(center, tvect)
        radius = diff
        if pointShape[1] == 2:
            array0, array1 = np.abs(vect[0, :]), np.abs(vect[1, :])
            index0, index1 = np.argmax(array0), np.argmax(array1)
            radius[index0], radius[index1] = diff[0], diff[1]
        if pointShape[1] == 3:
            array0, array1, array2 = np.abs(vect[0, :]), np.abs(vect[1, :]), np.abs(vect[2, :])
            index0, index1, index2 = np.argmax(array0), np.argmax(array1), np.argmax(array2)
            radius[index0], radius[index1], radius[index2] = diff[0], diff[1], diff[2]
        eigenvalue = v
        eigenvector = vect
        return corners
        # return corners, center, radius, eigenvalue, eigenvector


def min_volume_obb_corners_from_mesh(mesh_path=None, mesh=None, angle_digits=1, ordered=True):
    if mesh is None:
        if mesh_path is None:
            raise ValueError("Either mesh_path or mesh must be provided.")
        mesh = trimesh.load_mesh(mesh_path)
    to_origin, extents = trimesh.bounds.oriented_bounds(
        mesh,
        angle_digits=angle_digits,
        ordered=ordered,
    )
    half = np.asarray(extents, dtype=np.float64) * 0.5
    local = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [ half[0], -half[1], -half[2]],
            [ half[0],  half[1], -half[2]],
            [-half[0],  half[1], -half[2]],
            [-half[0], -half[1],  half[2]],
            [ half[0], -half[1],  half[2]],
            [ half[0],  half[1],  half[2]],
            [-half[0],  half[1],  half[2]],
        ],
        dtype=np.float64,
    )
    world = trimesh.transform_points(local, np.linalg.inv(to_origin))
    return world


def judge_rect(o: np.array, x: np.array, y: np.array, z: np.array):
    zero_like = 1e-7
    try:
        ox, oy, oz = x - o, y - o, z - o
        norm_ox, norm_oy, norm_oz = np.linalg.norm(ox), np.linalg.norm(oy), np.linalg.norm(oz)
        if norm_ox < zero_like or norm_oy < zero_like or norm_oz < zero_like:
            return False
        unit_ox, unit_oy, unit_oz = ox / norm_ox, oy / norm_oy, oz / norm_oz
        dot_tol = 1e-4
        if (
            abs(np.dot(unit_ox, unit_oy)) > dot_tol
            or abs(np.dot(unit_oy, unit_oz)) > dot_tol
            or abs(np.dot(unit_oz, unit_ox)) > dot_tol
        ):
            return False
        # right hand judge ixj=k, jxk=i, kxi=j,
        oxy, oyz, ozx = np.cross(ox, oy), np.cross(oy, oz), np.cross(oz, ox)
        xx, yy, zz = np.dot(oz, oxy), np.dot(oy, ozx), np.dot(ox, oyz)
        cxx, cyy, czz =  np.cross(oz, oxy), np.cross(oy, ozx), np.cross(ox, oyz)
        cxx, cyy, czz = np.abs(np.linalg.norm(cxx)), np.abs(np.linalg.norm(cyy)), np.abs(np.linalg.norm(czz))
        # print("xx, yy, zz:0 ", xx, yy, zz)

        return xx > 0 and yy > 0 and zz > 0 and cxx < zero_like and  cyy < zero_like and czz < zero_like # all are 90 and follow right hand rule
    except:
        print("err at judging")
        return False

def generate_full_rect(o: np.array, x: np.array, y: np.array, z: np.array):
    ox, oy, oz = x - o, y - o, z - o
    pxy = ox + oy + o
    pxz = ox + oz + o
    pyz = oy + oz + o
    pxyz = ox + oy + oz + o # other for points
    return [o, x, y, z, pxy, pxz, pyz, pxyz]

def generate_full_rect_using_rect(r: Rect):
    o, x, y, z = r.o, r.x, r.y, r.z
    ox, oy, oz = x - o, y - o, z - o
    pxy = ox + oy + o
    pxz = ox + oz + o
    pyz = oy + oz + o
    pxyz = ox + oy + oz + o # other for points
    return [o, x, pxy, y, z, pxz, pxyz, pyz] # x-first return for align

def sample_cuboid_surface(vertices, num_samples):
    # 计算长方体的六个面
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]]
    ]

    samples, random_uvs = [], []
    for face in faces:
        # 生成随机点在面上
        u = np.random.rand(num_samples)
        v = np.random.rand(num_samples)
        random_uvs.append([u, v])
        # 插值计算面上的坐标
        for i in range(num_samples):
            sample_point = (1 - u[i]) * (1 - v[i]) * face[0] + u[i] * (1 - v[i]) * face[1] + u[i] * v[i] * face[2] + (1 - u[i]) * v[i] * face[3]
            samples.append(sample_point)

    return np.array(samples), random_uvs

def corresponding_sample_points(rect: Rect, random_uvs):
    vertices = generate_full_rect_using_rect(rect)
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]]
    ]
    us = np.array(random_uvs)[:, 0, :]
    vs = np.array(random_uvs)[:, 1, :]
    # print("Array1 shape:", u.shape)
    # print("Array2 shape:", v.shape)
    samples = []
    for index in range(0, 6):
        face = faces[index]
        u = us[index]
        v = vs[index]
        num_samples = len(u)
        for i in range(num_samples):
            sample_point = (1 - u[i]) * (1 - v[i]) * face[0] + u[i] * (1 - v[i]) * face[1] + u[i] * v[i] * face[2] + (
                        1 - u[i]) * v[i] * face[3]
            samples.append(sample_point)
    return samples

def calc_sdf_diff(sdfs1, sdfs2):
    diff = sdfs1 - sdfs2
    diff_sum = np.sum(np.abs(diff))
    return diff, diff_sum

def calc_rect_ori_from_axis(x, y, z): # calc this RT from delta_x y z axis
    # 归一化处理
    x = x / np.linalg.norm(x)
    y = y / np.linalg.norm(y)
    z = z / np.linalg.norm(z)
    # 构建正交矩阵
    orthogonal_matrix = np.column_stack((x, y, z))
    # rotation_matrix = np.eye(4)
    # rotation_matrix[:3, :3] = orthogonal_matrix
    return orthogonal_matrix # this is R

def calc_rect_ori_from_rect(r: Rect): # calc this RT from
    o, x, y, z = r.o, r.x, r.y, r.z
    ox, oy, oz = x - o, y - o, z - o
    return calc_rect_ori_from_axis(ox, oy, oz)

#r1 is target r2 is source
def pairwise_obb_single(
        fr1,
        fr2,
        random_uvs,
        r1_SDF: Mesh2SDF,
        r2_SDF: Mesh2SDF,
        drop_worst_faces=0,
        return_facewise=False):
    """
    只进行一次obb配准和SDF计算，不做循环

    输入：
        fr1: full_rect1 (8x3 array)
        fr2: full_rect2 (8x3 array)
        random_uvs: 面上采样的随机uv
        r1_SDF: mesh1的Mesh2SDF
        r2_SDF: mesh2的Mesh2SDF

    返回：
        RT: 4x4 相对位姿
        diff_sum: SDF差值总和；drop_worst_faces > 0 时为trim后的面SDF总和
    """
    # step1: 生成rect
    tmp_rec2 = select_rect_from_full(fr2)
    r1 = select_rect_from_full(fr1)

    # step2: 采样
    static_samples = corresponding_sample_points(r1, random_uvs)
    tmp_samples = corresponding_sample_points(tmp_rec2, random_uvs)

    # step3: SDF计算
    fr1_sdfs = r1_SDF.calculate_sdf(static_samples)
    fr2_sdfs = r2_SDF.calculate_sdf(tmp_samples)
    sdf_diff, raw_diff_sum = calc_sdf_diff(fr1_sdfs, fr2_sdfs)
    samples_per_face = len(sdf_diff) // 6
    face_sdf_diffs = []
    for face_idx in range(6):
        start = face_idx * samples_per_face
        end = (face_idx + 1) * samples_per_face
        face_sdf_diffs.append(float(np.sum(np.abs(sdf_diff[start:end]))))
    drop_worst_faces = max(0, min(int(drop_worst_faces), 5))
    if drop_worst_faces > 0:
        kept_face_indices = np.argsort(face_sdf_diffs)[:6 - drop_worst_faces]
        diff_sum = float(np.sum(np.asarray(face_sdf_diffs)[kept_face_indices]))
    else:
        kept_face_indices = np.arange(6)
        diff_sum = float(raw_diff_sum)

    # step4: 求相对RT
    fr1_T, fr1_R = r1.o, calc_rect_ori_from_rect(r1)
    fr1_RT = np.eye(4)
    fr1_RT[:3, :3] = fr1_R
    fr1_RT[:3, 3] = fr1_T
    fr1_RT_inv = np.linalg.inv(fr1_RT)

    tmp_r2_T = tmp_rec2.o
    tmp_r2_R = calc_rect_ori_from_rect(tmp_rec2)
    tmp_r2_RT = np.eye(4)
    tmp_r2_RT[:3, :3] = tmp_r2_R
    tmp_r2_RT[:3, 3] = tmp_r2_T

    rel_RT = tmp_r2_RT @ fr1_RT_inv

    if return_facewise:
        facewise = {
            "raw_sdf_diff": float(raw_diff_sum),
            "trimmed_sdf_diff": float(diff_sum),
            "drop_worst_faces": int(drop_worst_faces),
            "kept_face_indices": [int(i) for i in kept_face_indices],
            "face_sdf_diffs": [float(v) for v in face_sdf_diffs],
        }
        return rel_RT, diff_sum, facewise
    return rel_RT, diff_sum
#r1 is target r2 is source
def pairwise_obb(r1: Rect, r2: Rect, r1_SDF: Mesh2SDF, r2_SDF: Mesh2SDF):
    fr1, fr2 = generate_full_rect_using_rect(r1), generate_full_rect_using_rect(r2)
    test_indexes = generate_enum_index(r2)
    static_samples, random_uvs = sample_cuboid_surface(fr1, num_samples=10000)
    raw_R, raw_T, raw_RT = None, None, None
    fr1_sdfs = r1_SDF.calculate_sdf(static_samples)
    fr1_T, fr1_R = r1.o, calc_rect_ori_from_rect(r1)
    fr1_RT = np.eye(4)
    fr1_RT[:3, :3] = fr1_R
    fr1_RT[:3, 3] = fr1_T
    fr1_RT_inv = np.linalg.inv(fr1_RT)
    record_sdf_diff, coff_index = 1e10, -1
    best_facewise_info = None

    faces_fr1 = [
        [fr1[0], fr1[1], fr1[2], fr1[3]],
        [fr1[4], fr1[5], fr1[6], fr1[7]],
        [fr1[0], fr1[1], fr1[5], fr1[4]],
        [fr1[2], fr1[3], fr1[7], fr1[6]],
        [fr1[0], fr1[3], fr1[7], fr1[4]],
        [fr1[1], fr1[2], fr1[6], fr1[5]],
    ]

    for index in range(0, 24):
        i2, j2, k2, l2 = test_indexes[index]
        tmp_rec2 = Rect(with_init=True, o=fr2[i2], x=fr2[j2], y=fr2[k2], z=fr2[l2])
        tmp_fr2 = generate_full_rect_using_rect(tmp_rec2)
        tmp_faces_fr2 = [
            [tmp_fr2[0], tmp_fr2[1], tmp_fr2[2], tmp_fr2[3]],
            [tmp_fr2[4], tmp_fr2[5], tmp_fr2[6], tmp_fr2[7]],
            [tmp_fr2[0], tmp_fr2[1], tmp_fr2[5], tmp_fr2[4]],
            [tmp_fr2[2], tmp_fr2[3], tmp_fr2[7], tmp_fr2[6]],
            [tmp_fr2[0], tmp_fr2[3], tmp_fr2[7], tmp_fr2[4]],
            [tmp_fr2[1], tmp_fr2[2], tmp_fr2[6], tmp_fr2[5]],
        ]

        tmp_samples = corresponding_sample_points(rect=tmp_rec2, random_uvs=random_uvs)
        fr2_sdfs = r2_SDF.calculate_sdf(tmp_samples)
        sdf_diff, diff_sum = calc_sdf_diff(fr1_sdfs, fr2_sdfs)
        print("SDF diff sum ", diff_sum)

        if diff_sum < record_sdf_diff:
            coff_index = index
            record_sdf_diff = diff_sum
            tmp_r2_T = tmp_rec2.o
            tmp_r2_R = calc_rect_ori_from_rect(tmp_rec2)
            tmp_r2_RT = np.eye(4)
            tmp_r2_RT[:3, :3] = tmp_r2_R
            tmp_r2_RT[:3, 3] = tmp_r2_T
            rel_mat_r12r2 = np.dot(tmp_r2_RT, fr1_RT_inv)
            raw_R = rel_mat_r12r2[:3, :3]
            raw_T = rel_mat_r12r2[:3, 3]
            raw_RT = np.eye(4)
            raw_RT[:3, :3] = raw_R
            raw_RT[:3, 3] = raw_T

            samples_per_face = len(tmp_samples) // 6
            facewise_info = []
            for i in range(6):
                start = i * samples_per_face
                end = (i + 1) * samples_per_face
                face_diff = np.sum(np.abs(fr1_sdfs[start:end] - fr2_sdfs[start:end]))
                face1_coords = np.array(faces_fr1[i])
                face2_coords = np.array(tmp_faces_fr2[i])
                face_info = {
                    "face1_coords": face1_coords,
                    "face2_coords": face2_coords,
                    "sdf_diff": face_diff
                }
                facewise_info.append(face_info)
            best_facewise_info = facewise_info
            print("upd min sdf error with idx ", best_facewise_info)
    if raw_R is None or raw_T is None:
        print("ERR at pairwise OBBs")
    else:
        print("most approximate index: ", coff_index)
    return raw_RT, record_sdf_diff, best_facewise_info

def get_cuboid_faces(vertices):
    return [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # Left
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
    ]

def face_to_rect_with_origin(face, origin_idx):
    o = face[origin_idx]
    # 定义面上的两个边向量，根据当前原点选择其余三个点中与其不共线的两个点构造坐标系
    other_indices = [i for i in range(4) if i != origin_idx]
    x_axis = face[other_indices[0]] - o
    y_axis = face[other_indices[1]] - o
    # 正交化处理
    z_axis = np.cross(x_axis, y_axis)
    if np.linalg.norm(z_axis) < 1e-6:
        return None
    return Rect(with_init=True, o=o, x=o + x_axis, y=o + y_axis, z=o + z_axis)
#r1 is target r2 is source
def pairwise_obb_face(r1: Rect, r2: Rect, r1_SDF: Mesh2SDF, r2_SDF: Mesh2SDF, num_samples=10000):
    fr1 = generate_full_rect_using_rect(r1)
    fr2 = generate_full_rect_using_rect(r2)

    faces2 = get_cuboid_faces(fr2)

    # 使用 r1 的第一个面作为采样基准面
    static_samples, random_uvs = sample_cuboid_surface(fr1, num_samples=num_samples)
    fr1_sdfs = r1_SDF.calculate_sdf(static_samples)

    fr1_T, fr1_R = r1.o, calc_rect_ori_from_rect(r1)
    fr1_RT = np.eye(4)
    fr1_RT[:3, :3] = fr1_R
    fr1_RT[:3, 3] = fr1_T
    fr1_RT_inv = np.linalg.inv(fr1_RT)

    best_diff, best_RT = 1e10, None

    for j, f2 in enumerate(faces2):
        for origin_idx in range(4):
            rect2_candidate = face_to_rect_with_origin(f2, origin_idx)
            if rect2_candidate is None:
                continue

            tmp_samples = corresponding_sample_points(rect2_candidate, random_uvs)
            fr2_sdfs = r2_SDF.calculate_sdf(tmp_samples)

            _, diff_sum = calc_sdf_diff(fr1_sdfs, fr2_sdfs)
            print(f"[ref face vs target face {j} origin {origin_idx}] sdf diff: {diff_sum:.4f}")

            if diff_sum < best_diff:
                best_diff = diff_sum

                tmp_r2_T = rect2_candidate.o
                tmp_r2_R = calc_rect_ori_from_rect(rect2_candidate)
                tmp_r2_RT = np.eye(4)
                tmp_r2_RT[:3, :3] = tmp_r2_R
                tmp_r2_RT[:3, 3] = tmp_r2_T

                rel_mat_r12r2 = np.dot(tmp_r2_RT, fr1_RT_inv)
                raw_R = rel_mat_r12r2[:3, :3]
                raw_T = rel_mat_r12r2[:3, 3]
                best_RT = np.eye(4)
                best_RT[:3, :3] = raw_R
                best_RT[:3, 3] = raw_T

    if best_RT is None:
        print("\u26a0\ufe0f 未找到合适的面配对")
    else:
        print(f"\u2705 最佳面配对 SDF diff: {best_diff:.6f}")
    return best_RT, best_diff

def pairwise_obb_plus(r1: Rect, r2: Rect, r1_SDF: Mesh2SDF, r2_SDF: Mesh2SDF, acc_flag=True): #
    fr1 = generate_full_rect_using_rect(r1) # full rect 1
    test_indexes = generate_enum_index(r1)
    raw_RT, record_sdf_diff = None, 1e10

    for index in range(0, 24): # in plus function, we also enum (24 ? 8) in rect 1 to find best suit obb
        if acc_flag:
            if index % 3 == 0:
                i1, j1, k1, l1 = test_indexes[index]
                tmp_rec1 =  Rect(with_init=True, o = fr1[i1], x = fr1[j1], y=fr1[k1], z=fr1[l1])
            else:
                continue
        else:
            i1, j1, k1, l1 = test_indexes[index]
            tmp_rec1 = Rect(with_init=True, o=fr1[i1], x=fr1[j1], y=fr1[k1], z=fr1[l1])
        tmp_raw_RT, tmp_diff, _ = pairwise_obb(r1 = tmp_rec1, r2 = r2, r1_SDF= r1_SDF, r2_SDF=r2_SDF)
        if tmp_diff < record_sdf_diff:
            print("renew rec_diff to ", tmp_diff)
            record_sdf_diff = tmp_diff
            raw_RT = tmp_raw_RT

    return raw_RT, record_sdf_diff


def has_duplicate(arr):
    unique_elements, counts = np.unique(arr, return_counts=True)
    if len(unique_elements) != len(arr):
        return True
    else:
        return False

def generate_enum_index(rect1: Rect):
    fr1 = generate_full_rect_using_rect(rect1)
    out_form = []
    for i in range(0, 8):
        for j in range(0, 8):
            for k in range(0, 8):
                for l in range(0, 8):
                    if has_duplicate(np.array([i, j, k, l])):
                        continue
                    else:
                        o, x, y, z = fr1[i], fr1[j], fr1[k], fr1[l]
                        if judge_rect(o, x, y, z):
                            # print("valid index seq: ", i, j, k, l)
                            out_form.append([i, j, k, l])
                        else:
                            continue
    return out_form

def select_rect_from_full(fr1): # fr1 is a full rect
    for i in range(0, 8):
        for j in range(0, 8):
            for k in range(0, 8):
                for l in range(0, 8):
                    if has_duplicate(np.array([i, j, k, l])):
                        continue
                    else:
                        o, x, y, z = fr1[i], fr1[j], fr1[k], fr1[l]
                        if judge_rect(o, x, y, z):
                            # print("valid index seq: ", i, j, k, l)
                            return Rect(with_init=True, o=o, x=x, y=y, z=z)
    # Fallback for ordered OBB corner sets such as trimesh.bounds.oriented_bounds:
    # [---, +--, ++-, -+-, --+, +-+, +++, -++] => o=0, x=1, y=3, z=4.
    # Some cropped meshes miss the strict judge tolerance above, but this
    # canonical ordering is still a valid cuboid frame.
    try:
        o, x, y, z = fr1[0], fr1[1], fr1[3], fr1[4]
        ox, oy, oz = x - o, y - o, z - o
        norm_x, norm_y, norm_z = np.linalg.norm(ox), np.linalg.norm(oy), np.linalg.norm(oz)
        if norm_x > 1e-9 and norm_y > 1e-9:
            if norm_z <= 1e-9:
                normal = np.cross(ox, oy)
                norm_normal = np.linalg.norm(normal)
                if norm_normal > 1e-12:
                    oz = normal / norm_normal * min(norm_x, norm_y)
                    z = o + oz
            print("WARN: select_rect_from_full fallback to canonical ordered OBB frame")
            return Rect(with_init=True, o=o, x=x, y=y, z=z)
    except Exception:
        pass
    try:
        points = np.asarray(fr1, dtype=np.float64)
        for i in range(len(points)):
            for j in range(len(points)):
                if j == i:
                    continue
                ox = points[j] - points[i]
                norm_x = np.linalg.norm(ox)
                if norm_x <= 1e-9:
                    continue
                for k in range(len(points)):
                    if k == i or k == j:
                        continue
                    oy = points[k] - points[i]
                    normal = np.cross(ox, oy)
                    norm_y = np.linalg.norm(oy)
                    norm_normal = np.linalg.norm(normal)
                    if norm_y <= 1e-9 or norm_normal <= 1e-12:
                        continue
                    z = points[i] + normal / norm_normal * min(norm_x, norm_y)
                    print("WARN: select_rect_from_full fallback to non-collinear planar OBB frame")
                    return Rect(with_init=True, o=points[i], x=points[j], y=points[k], z=z)
    except Exception:
        pass
    print("ERR for this rec! ")
    return None

def evaluate_extrinsic_difference_OBB(
    M1: np.ndarray,
    M2: np.ndarray,
    rotation_threshold_deg: float = 5.0,
    translation_threshold: float = 0.01,
    return_all: bool = False
):
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
    返回:
    -------
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
    #    旋转矩阵 R 的旋转角度可通过以下公式计算：angle = arccos((trace(R) - 1) / 2)
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
    # print("rotation diff: ", rotation_angle_deg)
    # print("translation diff ", translation_diff)
    if return_all:
        return rotation_ok and translation_ok, rotation_angle_deg, translation_diff
    else:
        return rotation_ok and translation_ok

def visualize_facewise_sdf_diff(facewise_info):

    vis_list = []
    sdf_diffs = [face["sdf_diff"] for face in facewise_info]
    max_diff = max(sdf_diffs)
    norm_diffs = [diff / max_diff for diff in sdf_diffs] if max_diff > 0 else [0] * len(sdf_diffs)

    for i, face in enumerate(facewise_info):
        color = plt.cm.viridis(norm_diffs[i])[:3]  # RGB 归一化颜色

        for key in ["face1_coords", "face2_coords"]:
            coords = face[key]
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(coords)
            mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
            mesh.paint_uniform_color(color)
            mesh.compute_vertex_normals()
            vis_list.append(mesh)

    print("Showing face-by-face sdf error visualization...")
    o3d.visualization.draw_geometries(vis_list)
def save_rays_as_ply(rays_o, rays_d, filename, scale=0.2):
    """
    Save rays as line segments to a PLY file using Open3D.
    rays_o: [N, 3] tensor of ray origins
    rays_d: [N, 3] tensor of ray directions
    filename: path to save .ply
    scale: length of each ray line segment
    """
    rays_o_np = rays_o.detach().cpu().numpy()
    rays_d_np = rays_d.detach().cpu().numpy()

    # Compute end points
    rays_end = rays_o_np + rays_d_np * scale

    # Combine all points (start and end of each ray)
    points = []
    lines = []
    for i in range(len(rays_o_np)):
        start = rays_o_np[i]
        end = rays_end[i]
        points.append(start)
        points.append(end)
        lines.append([2*i, 2*i + 1])

    # Create Open3D geometry
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    # (Optional) assign color to all rays
    colors = [[1, 0, 0] for _ in lines]  # Red
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Save to file
    o3d.io.write_line_set(filename, line_set)
    print(f"Saved rays to {filename}")


if __name__ == '__main__':
    # o = np.array([0, 0, 0])
    # z = np.array([0, 0, 1])
    # y = np.array([0, 1, 0])
    # x = np.array([1, 0, 0])
    # r1 = Rect(with_init=True, o=o, x=x, y=y, z=z)
    # generate_enum_index(r1)
    mesh_path_source, mesh_path_target = './logs/test/bunny_pose1.ply',  './logs/test/bunny_pose2.ply'
    mesh_path_source, mesh_path_target = './logs/test/02_exp_final_mesh.ply',  './logs/test/01_exp_final_mesh.ply'

    # mesh_path_source, mesh_path_target= 'C:/Users/guanl/Desktop/reg/exp/01_exp_final_mesh.ply',  'C:/Users/guanl/Desktop/reg/exp/03_exp_final_mesh.ply' #default

    target_sdf, source_sdf = Mesh2SDF(mesh_path=mesh_path_target), Mesh2SDF(mesh_path=mesh_path_source)
    full_rect1, full_rect2 = target_sdf.calculate_obb_from_point_cloud(), source_sdf.calculate_obb_from_point_cloud()
    rect1, rect2 = select_rect_from_full(full_rect1), select_rect_from_full(full_rect2) # simplify OBB as o x y z
    # raw_RT, _ = pairwise_obb(r1 = rect1, r2 = rect2, r1_SDF = target_sdf, r2_SDF=source_sdf)
    raw_RT, _, best_pairwise_info = pairwise_obb(r1 = rect1, r2 = rect2, r1_SDF = target_sdf, r2_SDF=source_sdf)
    visualize_facewise_sdf_diff(best_pairwise_info)
    # raw_RT, _ = pairwise_obb_plus(r1 = rect1, r2 = rect2, r1_SDF = sdf1, r2_SDF=sdf2, acc_flag=False)

    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(full_rect1)
    # o3d.io.write_point_cloud('C:/Users/guanli.hou/Desktop/rt_60.ply', point_cloud)
    # point_cloud.points = o3d.utility.Vector3dVector(full_rect2)
    # o3d.io.write_point_cloud('C:/Users/guanli.hou/Desktop/stand.ply', point_cloud)
    print(raw_RT)
    # print(raw_T)

    # res = judge_rect(o, x, y, z)
    # print(res)
    # res = judge_rect(o, x, y, -z)
    # print(res)
    M1_example = [[0.7423551082611084, -0.2518492341041565, -0.6208710670471191, 0.49617502093315125], [-0.6630696654319763, -0.14315412938594818, -0.7347418069839478, 0.22048130631446838], [0.09616389870643616, 0.9571200609207153, -0.2732647955417633, 0.20496077835559845], [0.0, 0.0, 0.0, 1.0]]

    M2_example = [[-0.9261959195137024, 0.004189362749457359, -0.37701937556266785, 0.2073606550693512], [0.37704265117645264, 0.010291064158082008, -0.926138699054718, 0.5093762874603271], [-2.3283061589829401e-10, -0.9999382495880127, -0.011111109517514706, 0.0061111110262572765], [0.0, 0.0, 0.0, 1.0]]

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
        ]) # a test for obb based exc finding
    M1_example = RT @ M1_example # trans with raw RT
    result = evaluate_extrinsic_difference(
        M1_example,
        M2_example,
        rotation_threshold_deg=30.0,
        translation_threshold=0.5
    )
    print("is differ:", result)


import math
import numpy as np


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def get_intrinsic_matrix(focal, width, height):
    # camera looks at -z, x right, y down, origin at top-left
    K = np.array([
        [focal, 0, -width / 2],
        [0, focal, -height / 2],
        [0, 0, -1]
    ])
    K = np.array([
        [1, 0, 0],
        [0, 1, height],
        [0, 0, 1]
    ]) @ np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ]) @ K
    return K


def get_projection_matrix(K, c2w):
    # K: 3x3
    # c2w: 3x4 or 4x4, camera to world, x right, y up, z backward
    if c2w.shape[0] == 3:
        c2w = np.concatenate([c2w, np.array([[0, 0, 0, 1]])], axis=0)
    w2c = np.linalg.inv(c2w)
    P = K @ w2c[:3, :]
    return P


def project_points(points, P):
    # points: Nx3
    # P: 3x4
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points = points @ P.T
    points = points / points[:, 2:]
    return points[:, :2]


def project_sphere(center, radius, pose, K):
    # center: 3
    # radius: 1
    # P: 3x4
    P = get_projection_matrix(K, pose)
    center2d = project_points(center[None], P)[0]

    if radius is None:
        return center2d, None
    x_dir = pose[:3, 0]
    x_dir = x_dir / np.linalg.norm(x_dir) # pose may contain scaling
    r_point = center + x_dir * radius
    r_point2d = project_points(r_point[None], P)
    radius2d = np.linalg.norm(center2d[None] - r_point2d, axis=1).item()
    radius2d = round(radius2d)
    return center2d, radius2d


if __name__ == "__main__":
    pose = np.eye(4)
    focal, width, height = 128, 256, 256
    points3d = np.array([[0, 0, -1], [0, 0, -2], [1, 1, -1], [1, 0, -1], [-1, 1, -1], [1, -1, -1]])
    K = get_intrinsic_matrix(focal, width, height)

    P = get_projection_matrix(K, pose)

    points = project_points(points3d, P)
    print(points)

    print("project sphere:")
    for i in range(points3d.shape[0]):
        centers = project_sphere(points3d[i], None, pose, K)
        print(centers)

from typing import Tuple, Union, Optional, List, Dict
import torch
import torch.nn.functional as F
import numpy as np
import os
import imageio
import skimage
from PIL import Image
# BLENDER CONVENSION for environment map
# canonical orientation : from -x looking at origin (+x direction)
# envmap, theta = 0 ~ pi, phi = pi ~ -pi:
# 0/pi -----------------> -pi
# |                   .
# -x   +y   +x   -y   -x  (directions)
# |                   . 
# pi ..................

canonical_camera_orientation = torch.tensor([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0]
]).float()
canonical_camera_zdir = torch.tensor([-1, 0, 0]).float()


def create_envmap_grid(size: int):
    """
    BLENDER CONVENSION
    Create the grid of environment map that contain the position in sperical coordinate
    Top left is (0,pi) and bottom right is (pi, -pi), indexing 'xy'
    """    
    thetas = torch.linspace(0, torch.pi, size) # vertical, 0 ~ pi
    phis = torch.linspace(torch.pi, -torch.pi, 2 * size) # horizontal, pi ~ -pi
    thetas, phis = torch.meshgrid(thetas, phis, indexing='ij')
    theta_phi = torch.stack([thetas, phis], dim=-1)
    return theta_phi

def envmap_grid_to_cartesian(theta_phi: torch.Tensor):
    """
    BLENDER CONVENSION
    Convert the grid of environment map to cartesian coordinates
    theta_phi: [H, W, 2], spherical coordinates
    """
    theta, phi = theta_phi[..., 0], theta_phi[..., 1]
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)

def get_normal_vector(incoming_vector: torch.Tensor, reflect_vector: torch.Tensor):
    """
    BLENDER CONVENSION
    incoming_vector: the vector from the point to the camera
    reflect_vector: the vector from the point to the light source
    """
    #N = 2(R ⋅ I)R - I
    N = (incoming_vector + reflect_vector) / torch.linalg.norm(incoming_vector + reflect_vector, dim=-1, keepdims=True)
    return N


def get_cartesian_from_spherical(theta: torch.Tensor, phi: torch.Tensor, r = 1.0):
    """
    BLENDER CONVENSION
    theta: vertical angle
    phi: horizontal angle
    r: radius
    """
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.stack([x,y,z], axis=-1)


def get_spherical_from_cartesian(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, to_2pi=True):
    """
    theta: vertical angle
    phi: horizontal angle
    """
    r = torch.linalg.norm(torch.stack([x,y,z], axis=-1), dim=-1)
    theta = torch.acos(z / r) # [0, pi]
    phi = torch.atan2(y, x)
    if to_2pi:
        phi = (phi + 2 * torch.pi) % (2 * torch.pi) # [0, 2pi]
    return theta, phi, r
    
'''
def ballimg2envmap(ball_img: torch.Tensor, envmap_size: int, pose: torch.tensor = None, interp_mode='bilinear'):
    """Reconstruct the envmap from a refective ball.
        ball_img: [B, C, H, W], a rendered ball image reflecting the envmap
        pose: [B, 3, 4], camera-to-world matrix, opengl convention
    """
    I = canonical_camera_zdir.to(ball_img)
    env_grid = create_envmap_grid(envmap_size).to(ball_img)
    thetas, phis = env_grid[...,0], env_grid[...,1]
    reflect_vec = get_cartesian_from_spherical(thetas, phis)
    normal = get_normal_vector(I[None,None], reflect_vec)
    
    # turn from normal map to position to lookup
    # using pytorch method for bilinear interpolation
    pos = normal[..., 1:] * -1
    grid = pos[None].expand(ball_img.shape[0], -1, -1, -1) # [B, H, W, 2]

    envmap = torch.nn.functional.grid_sample(ball_img, grid, mode=interp_mode, padding_mode='border', align_corners=True)

    if pose is not None:
        orientation = pose[:, :3, :3]
        # apply world-to-cam1, cam2-to-world
        rotation = orientation @ torch.linalg.inv(canonical_camera_orientation.to(pose).float()).to(pose)
        envmap = rotate_envmap(envmap, rotation, interp_mode)
    return envmap
'''

def envmap2ballimg(envmap: torch.tensor, ballimg_size: int, pose: torch.tensor = None, interp_mode='bilinear'):
    """Render a reflective ball image from camera poses.
        envmap: [B, C, H, W], a canonical envmap, i.e., center corresponds +x
        pose: [B, 3, 4], camera-to-world matrix, opengl convention
    """
    if pose is not None:
        orientation = pose[:, :3, :3]
        # apply world-to-cam1, cam2-to-world, then take the inverse rotation
        rotation = torch.linalg.inv(
            orientation.float() @ torch.linalg.inv(canonical_camera_orientation.to(pose).float())
        ).to(pose)
        envmap = rotate_envmap(envmap, rotation, interp_mode)

    z = torch.linspace(1, -1, ballimg_size).to(envmap)
    y = torch.linspace(1, -1, ballimg_size).to(envmap)
    z, y = torch.meshgrid(z, y, indexing='ij')
    x2 = (1 - y**2 - z**2)
    mask = x2 >= 0
    x = -torch.clip(x2, 0, 1).sqrt()

    normal = torch.stack([x, y, z], dim=-1) # [H, W, 3]
    normal = normal / torch.linalg.norm(normal, dim=-1, keepdims=True)

    I = canonical_camera_zdir.to(envmap)
    reflect_vec = 2 * torch.sum(normal * I, dim=-1, keepdims=True) * normal - I
    # reflect_vec = reflect_vec / torch.linalg.norm(reflect_vec, dim=-1, keepdims=True)
    
    thetas, phis, rs = get_spherical_from_cartesian(reflect_vec[...,0], reflect_vec[...,1], reflect_vec[...,2], to_2pi=False)
    thetas = thetas / (torch.pi / 2) - 1 # [0, pi] -> [-1, 1]
    phis = -phis / torch.pi  # [pi, -pi] -> [-1, 1]
    grid = torch.stack([thetas, phis], dim=-1).unsqueeze(0).expand(envmap.shape[0], -1, -1, -1) # [B, H, W, 2]

    ball_img = torch.nn.functional.grid_sample(envmap, grid.flip(-1), mode=interp_mode, padding_mode='border', align_corners=True)
    ball_img[~mask.expand_as(ball_img)] = 0

    return ball_img


import torch
import math

def fill_by_angle_nearest(
    envmap: torch.Tensor,        # [C, H, W]
    mask_nonzero: torch.Tensor,  # [H, W], 已填充像素=1, 未填充=0
    to_3d_fn,                    # 一个函数: (y, x) -> reflect_dir  (返回 3D 向量)
) -> torch.Tensor:
    """
    对 envmap 中尚未填充的像素，用“角度上最近邻”的已填充像素颜色来覆盖。
    to_3d_fn: 给定 envmap 像素 (y, x), 计算其在 3D 中的方向向量 (reflect_dir).
    
    envmap: [C, H, W]
    mask_nonzero: [H, W]
    返回 envmap_out: [C, H, W], 所有像素都被赋值。
    """
    C, H, W = envmap.shape
    device = envmap.device

    # 1. 收集已填充像素的 (dir, color)
    filled_coords = mask_nonzero.nonzero(as_tuple=False)  # shape [M, 2], (y, x)
    # 准备存放已填充的 3D 方向 & 颜色
    filled_dirs = []
    filled_colors = []

    for i in range(filled_coords.shape[0]):
        yy, xx = filled_coords[i]
        dir3d = to_3d_fn(yy.item(), xx.item())
        # 变成 torch.Tensor, 并归一化(避免数值误差)
        dir3d = dir3d / torch.norm(dir3d, p=2)
        filled_dirs.append(dir3d)
        # 拿到对应颜色
        col = envmap[:, yy, xx]
        filled_colors.append(col)

    # 转成张量方便后续批量运算 (optional)
    filled_dirs = torch.stack(filled_dirs, dim=0)     # [M, 3]
    filled_colors = torch.stack(filled_colors, dim=0) # [M, C]

    # 2. 遍历未填充像素, 找到角度最近邻
    envmap_out = envmap.clone()
    unfilled_coords = (mask_nonzero==0).nonzero(as_tuple=False)  # [N_u, 2]
    for i in range(unfilled_coords.shape[0]):
        yy, xx = unfilled_coords[i]
        dir3d_u = to_3d_fn(yy.item(), xx.item())  # shape [3]
        dir3d_u = dir3d_u / torch.norm(dir3d_u, p=2)  # 归一化

        # 2.1 计算与 filled_dirs 的点积
        # dot(u, v) = cos( angle ),  max(dot) => min(angle)
        # shape [M, 3] · [3] -> [M]
        dots = torch.sum(filled_dirs * dir3d_u, dim=-1)
        # 找到最大 dot => 角度最小
        best_idx = torch.argmax(dots)
        envmap_out[:, yy, xx] = filled_colors[best_idx]

    return envmap_out


def ballimg2envmap(
    ball_img: torch.Tensor,   # [C, H, W]
    ball_params: list[float], # [x, y, z, r]
    pose: torch.Tensor,       # [3, 4]
    K: torch.Tensor,          # [3, 3]
    image_height: int,
    image_width: int,
    envmap_height: int,
    envmap_width: int,
    fill_color: float = 0.0
) -> torch.Tensor:
    """
    1) 与普通 splat 逻辑相同: "球面图 -> envmap"
    2) 合成 envmap2 后, 对空洞 (mask_nonzero==0) 用“角度最近邻”填充
    """
    device = ball_img.device
    dtype = ball_img.dtype
    C = ball_img.shape[0]

    # (1) 初始化
    envmap_acc = torch.full((C, envmap_height, envmap_width), fill_color, 
                            device=device, dtype=dtype)
    envmap_weight = torch.zeros((envmap_height, envmap_width),
                                device=device, dtype=dtype)

    # (2) 生成图像平面像素坐标
    pixel_xs = torch.arange(0, image_width, device=device, dtype=dtype)
    pixel_ys = torch.arange(0, image_height, device=device, dtype=dtype)
    pixel_xs, pixel_ys = torch.meshgrid(pixel_xs, pixel_ys, indexing='xy')

    # (3) 射线
    ray_o_world, ray_d_world = pixels_to_rays(pixel_xs, pixel_ys, K, pose)
    ray_d_world = ray_d_world / torch.linalg.norm(ray_d_world, dim=-1, keepdims=True)

    # (4) 球面交点
    x, y, z, r = ball_params
    ball_center = torch.tensor([x, y, z], device=device, dtype=dtype)
    ball_center = ball_center.expand_as(ray_o_world)

    t = torch.sum((ball_center - ray_o_world) * ray_d_world, dim=-1, keepdim=True)
    nearest = ray_o_world + t * ray_d_world
    ball2rays_dists = torch.linalg.norm(nearest - ball_center, dim=-1, keepdim=True)
    ball_mask = (ball2rays_dists <= r).squeeze(-1)

    delta_t = (r**2 - ball2rays_dists**2).clamp_min(0).sqrt()
    hit = ray_o_world + (t - delta_t) * ray_d_world  # [H, W, 3]
    normal_world = (hit - ball_center) / r

    # (5) 反射方向
    rd = ray_d_world[ball_mask]
    n  = normal_world[ball_mask]
    reflect_vec = 2*torch.sum(n * -rd, dim=-1, keepdims=True)*n - -rd  # [N,3]

    # (6) 映射到 envmap (thetas, phis) -> (u,v)
    thetas, phis, _ = get_spherical_from_cartesian(
        reflect_vec[...,0], reflect_vec[...,1], reflect_vec[...,2], to_2pi=False
    )
    thetas = thetas / (torch.pi/2) - 1
    phis   = -phis / torch.pi

    u = (phis + 1) / 2 * (envmap_width  - 1)
    v = (thetas + 1) / 2 * (envmap_height - 1)

    ball_colors = ball_img.permute(1,2,0)[ball_mask]  # [N, C]

    # 双线性 splat
    u_floor = torch.floor(u)
    v_floor = torch.floor(v)
    u_ceil  = u_floor + 1
    v_ceil  = v_floor + 1

    alpha = u - u_floor
    beta  = v - v_floor

    w_ff = (1 - alpha)*(1 - beta)
    w_fc = (1 - alpha)*beta
    w_cf = alpha*(1 - beta)
    w_cc = alpha*beta

    u_floor = u_floor.long()
    v_floor = v_floor.long()
    u_ceil  = u_ceil.long()
    v_ceil  = v_ceil.long()

    def in_range(x, lim):
        return (x >= 0) & (x < lim)

    valid_ff = in_range(u_floor, envmap_width) & in_range(v_floor, envmap_height)
    valid_fc = in_range(u_floor, envmap_width) & in_range(v_ceil,  envmap_height)
    valid_cf = in_range(u_ceil,  envmap_width) & in_range(v_floor, envmap_height)
    valid_cc = in_range(u_ceil,  envmap_width) & in_range(v_ceil,  envmap_height)

    def scatter_add_splat(u_idx, v_idx, w, color, valid_mask):
        idx = valid_mask.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return

        u_val = u_idx[idx]
        v_val = v_idx[idx]
        w_val = w[idx]
        c_val = color[idx]

        gather_idx = v_val * envmap_width + u_val
        for c_ in range(C):
            envmap_acc_channel = envmap_acc[c_].view(-1)
            envmap_acc_channel.index_add_(0, gather_idx, w_val * c_val[:, c_])

        envmap_weight_flat = envmap_weight.view(-1)
        envmap_weight_flat.index_add_(0, gather_idx, w_val)

    scatter_add_splat(u_floor, v_floor, w_ff, ball_colors, valid_ff)
    scatter_add_splat(u_floor, v_ceil,  w_fc, ball_colors, valid_fc)
    scatter_add_splat(u_ceil,  v_floor, w_cf, ball_colors, valid_cf)
    scatter_add_splat(u_ceil,  v_ceil,  w_cc, ball_colors, valid_cc)

    # 合并
    envmap2 = envmap_acc.clone()
    envmap_weight_2d = envmap_weight.view(envmap_height, envmap_width)
    mask_nonzero = (envmap_weight_2d > 1e-8)
    for c_ in range(C):
        envmap2[c_][mask_nonzero] = envmap_acc[c_][mask_nonzero] / envmap_weight_2d[mask_nonzero]

    # 根据角度做最近邻填补
    def envmap_to_3d_dir(y, x):
        """
        将 envmap 像素 (y, x) -> 其 3D 方向.
        注意: 我们要逆着 (thetas, phis) 的映射来写.
        (y in [0, H], x in [0, W]) => thetas, phis => reflect_dir
        """
        # x => phis in [-1,1], y => thetas in [-1,1]
        phis_val = ( x / (envmap_width  - 1) )*2 - 1
        thetas_val = ( y / (envmap_height - 1) )*2 - 1

        # 反推 thetas, phis
        # thetas = [0, pi] => we mapped -> [-1,1], so thetas = (val+1)* (pi/2)
        # phis   = [-pi, pi] => we mapped -> [-1,1], so phis = - val * pi
        theta_ = (thetas_val + 1) * (math.pi/2)
        phi_   = - phis_val * math.pi

        # 再把 (theta_, phi_) 转回 3D direction
        # 这跟 get_cartesian_from_spherical() 相同, 
        # 需注意之前的定义: 
        #   get_cartesian_from_spherical(theta, phi)
        #   x = sin(theta)*cos(phi), y = sin(theta)*sin(phi), z = cos(theta)
        # 可能需跟你真正使用的 spherical <-> cartesian 对齐
        # 这里假设和 envmap2ballimg 一致:
        dir_x = math.sin(theta_)*math.cos(phi_)
        dir_y = math.sin(theta_)*math.sin(phi_)
        dir_z = math.cos(theta_)
        return torch.tensor([dir_x, dir_y, dir_z], device=device, dtype=dtype)

    envmap2_filled = fill_by_angle_nearest(envmap2, mask_nonzero, lambda yy, xx: envmap_to_3d_dir(yy, xx))
    return envmap2_filled


def envmap2ballimg_perspective(
    envmap: torch.tensor, ball_params: List, pose: torch.Tensor, K: torch.Tensor,
    image_height: int = None, image_width: int = None, interp_mode='bilinear',
    orthographics=False
):
    """Render a reflective ball image from camera poses.
        envmap: [C, H, W], a canonical envmap, i.e., center corresponds +x
        ball_params: [x, y, z, r], ball center and radius in 3D world space
        pose: [3, 4], camera-to-world matrix, opengl convention, may contain scaling...
        K: [3, 3], intrinsic matrix
    """
    device, dtype = pose.device, pose.dtype

    x, y, z, r = ball_params
    ball_center = torch.tensor([x, y, z], device=device, dtype=dtype)

    # pixels to rays in world space
    pixel_xs = torch.arange(0, image_width, device=device, dtype=dtype)
    pixel_ys = torch.arange(0, image_height, device=device, dtype=dtype)
    pixel_xs, pixel_ys = torch.meshgrid(pixel_xs, pixel_ys, indexing='xy')

    if orthographics:
        raise NotImplementedError("orthographics is not implemented")
        ray_d_world = torch.tensor([0, 0, 1.0]).expand(list(pixel_xs.shape) + [3]).to(pose)
        ray_d_world = torch.matmul(pose[:3, :3], ray_d_world[..., None])[..., 0]
    else:
        ray_o_world, ray_d_world = pixels_to_rays(pixel_xs, pixel_ys, K, pose)
    ray_d_world = ray_d_world / torch.linalg.norm(ray_d_world, dim=-1, keepdims=True) # NOTE: important since pose may contain scaling

    # ball to rays in world space
    ball_center = ball_center.expand_as(ray_o_world)
    t = torch.sum((ball_center - ray_o_world) * ray_d_world, dim=-1, keepdim=True)
    nearest = ray_o_world + t * ray_d_world
    # ball2rays = nearest - ball_center
    ball2rays_dists = torch.linalg.norm(nearest - ball_center, dim=-1, keepdim=True) # [H, W, 1]
    ball_mask = (ball2rays_dists <= r).squeeze(-1) # [H, W]

    # normal_world = ball2rays / ball2rays_dists
    delta_t = (r ** 2 - ball2rays_dists ** 2).clamp(0, None).sqrt() # [H, W, 1]
    hit = ray_o_world + (t - delta_t) * ray_d_world# [H, W, 3]
    normal_world = (hit - ball_center) / r # [H, W, 3]

    # mask out the pixels outside the ball
    ray_d_world = ray_d_world[ball_mask] # [N, 3]
    normal_world = normal_world[ball_mask] # [N, 3]

    # perfect reflection
    reflect_vec = 2 * torch.sum(normal_world * -ray_d_world, dim=-1, keepdims=True) * normal_world - -ray_d_world # [N, 3]

    thetas, phis, rs = get_spherical_from_cartesian(
        reflect_vec[...,0], reflect_vec[...,1], reflect_vec[...,2], to_2pi=False
    )

    thetas = thetas / (torch.pi / 2) - 1 # [0, pi] -> [-1, 1]
    phis = -phis / torch.pi  # [pi, -pi] -> [-1, 1]
    grid = torch.stack([thetas, phis], dim=-1).unsqueeze(0).unsqueeze(0) # [1, 1, N, 2]

    queried = torch.nn.functional.grid_sample(
        envmap.unsqueeze(0), grid.flip(-1), mode=interp_mode, padding_mode='border', align_corners=True
    ).squeeze(0).squeeze(1).transpose(1, 0) # [N, C]


    ball_img = torch.zeros((image_height, image_width, envmap.shape[0]), device=device, dtype=dtype)
    ball_img[ball_mask] = queried
    ball_img = ball_img.permute(2, 0, 1)


    return ball_img

def prepare_depth_control_image(depth_estimator, image, mask, depth_mode="local", depth=0):
    depth_map = depth_estimator(image)['predicted_depth']
    W, H = image.size

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(H, W),
        mode="bicubic",
        align_corners=False,
    )

    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)

    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    
    
    image = torch.cat([depth_map] * 3, dim=1)
    
    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]

    mask_bool = np.array(mask).astype(bool)
    if depth_mode == "local":
        #image[mask_bool] = np.max(image[mask_bool])

        #print(np.max(image[mask_bool]))

        image[mask_bool] = np.power(float(1.0)/depth, 1/2.3)

        
    elif depth_mode == "nearest":
        image[mask_bool] = 1
    elif depth_mode == "middle":
        image[mask_bool] = (1 + np.max(image[mask_bool])) / 2
    elif depth_mode == "farthest":
        image = np.percentile(image, 5)
    else:
        raise ValueError(f"Invalid depth_mode: {depth_mode}")

    image = Image.fromarray(skimage.img_as_ubyte(image))
    
    return image

def ballimg2envmap_perspective(
    ballimg: torch.tensor, 
    ball_params: List, 
    pose: torch.Tensor, 
    K: torch.Tensor,
    envmap_size: int, 
    interp_mode='bilinear', 
    orthographics=False
):
    """Render a reflective ball image from camera poses.
    
    Args:
        ballimg: [..., C, H, W], input ball image
        ball_params: [x, y, z, r], ball center and radius in 3D world space
        pose: [3, 4], camera-to-world matrix, opengl convention
        K: [3, 3], intrinsic matrix
        envmap_size: output environment map size
    """
    device, dtype = pose.device, pose.dtype
    
    # 处理输入图像维度
    if ballimg.dim() == 5:  # [B, N, C, H, W]
        B, N = ballimg.shape[:2]
        ballimg = ballimg.reshape(-1, *ballimg.shape[2:])  # [B*N, C, H, W]
    else:
        B, N = 1, 1
        if ballimg.dim() == 3:  # [C, H, W]
            ballimg = ballimg.unsqueeze(0)  # [1, C, H, W]
    
    image_height, image_width = ballimg.shape[-2:]
    
    # 创建环境图采样网格
    env_grid = create_envmap_grid(envmap_size).to(ballimg)
    thetas, phis = env_grid[...,0], env_grid[...,1]
    reflect_vec = get_cartesian_from_spherical(thetas, phis)
    
    # 获取球体属性
    ball_center = torch.tensor(ball_params[:3], device=device, dtype=dtype)
    ball_r = ball_params[3]
    cam_origin = pose[:3, 3]
    
    # 计算方向
    ball2cam_dir = cam_origin - ball_center
    ball2cam_dir = ball2cam_dir / torch.linalg.norm(ball2cam_dir)
    normal_dir = get_normal_vector(ball2cam_dir[None, None], reflect_vec)
    
    # 获取球面上的点
    points_on_ball = ball_center[None, None] + ball_r * normal_dir
    shape = points_on_ball.shape[:-1]
    
    # 投影点到图像空间
    P = get_projection_matrix(K, pose)
    pixels = project_points(points_on_ball.reshape(-1, 3), P).reshape(shape + (2,))
    
    # 归一化坐标到 [-1, 1] 范围
    pixels[..., 0] = (pixels[..., 0] + 0.5) / image_width * 2 - 1
    pixels[..., 1] = (pixels[..., 1] + 0.5) / image_height * 2 - 1
    
    # 重复采样网格以匹配输入维度
    pixels = pixels.unsqueeze(0)  # [1, envmap_size, envmap_size, 2]
    pixels = pixels.expand(B*N, -1, -1, -1)  # [B*N, envmap_size, envmap_size, 2]
    
    # grid_sample 期望输入为 [N, C, H, W] 和 grid 为 [N, H', W', 2]
    envmap = torch.nn.functional.grid_sample(
        ballimg, 
        pixels,
        mode=interp_mode, 
        padding_mode='border', 
        align_corners=True
    )
    
    # 恢复原始批次维度
    if B*N > 1:
        envmap = envmap.reshape(B, N, *envmap.shape[1:])
    else:
        envmap = envmap.squeeze(0)
    
    return envmap

def get_projection_matrix(K, c2w):
    # K: 3x3
    # c2w: 3x4 or 4x4, camera to world, x right, y up, z backward
    if c2w.shape[0] == 3:
        c2w = torch.concatenate([c2w, torch.tensor([[0, 0, 0, 1]]).to(c2w)], axis=0)
    w2c = torch.linalg.inv(c2w)
    P = K @ w2c[:3, :]
    return P


def project_points(points, P):
    # points: Nx3
    # P: 3x4
    points = torch.concatenate([points, torch.ones((points.shape[0], 1)).to(points)], axis=1)
    points = points @ P.T
    points = points / points[:, 2:]
    return points[:, :2]



def imagepoint2world(image_point, pose, K):
    x, y = image_point
    x = int(x)
    y = int(y)
    
    # 确保 depth_map 是 NumPy 数组
    #if isinstance(depth_map, torch.Tensor):
    #    depth_map = depth_map.cpu().numpy()
    #fixme: using a fixed close depth
    #depth = depth_map[y, x]
    depth = 4.0
    

    # 确保 K 是 NumPy 数组
    if isinstance(K, torch.Tensor):
        K = K.cpu().numpy()
    
    K_inv = np.linalg.inv(K)
    pixel_coords = np.array([x, y, 1])
    camera_coords = depth * K_inv @ pixel_coords
    
    # 确保 pose 是 NumPy 数组
    if isinstance(pose, torch.Tensor):
        pose = pose.cpu().numpy()
    
    world_coords = pose[:3, :3] @ camera_coords + pose[:3, 3]
    
    return world_coords

def imagepoint2worldwithdepth(image_point, pose, K, depth_queried):
    x, y = image_point
    x = int(x)
    y = int(y)
    

    depth = depth_queried
    
    # 确保 K 是 NumPy 数组
    if isinstance(K, torch.Tensor):
        K = K.cpu().numpy()
    
    K_inv = np.linalg.inv(K)
    pixel_coords = np.array([x, y, 1])
    camera_coords = depth * K_inv @ pixel_coords
    
    # 确保 pose 是 NumPy 数组
    if isinstance(pose, torch.Tensor):
        pose = pose.cpu().numpy()
    
    world_coords = pose[:3, :3] @ camera_coords + pose[:3, 3]
    
    return world_coords

def world2imagepoint(world_point, depth_map, pose, K):
    # input: world_point: (x, y, z), a single 3d point in camera coordinates
    # input: depth_map: (h, w), the depth map
    # input: pose: (4, 4), the camera pose matrix
    # input: K: (3, 3), the camera intrinsic matrix
    # output: (x, y), the image point in pixel coordinates
    world_point = np.append(world_point, 1)
    camera_coords = np.linalg.inv(pose) @ world_point
    pixel_coords = K @ camera_coords[:3]
    x, y = pixel_coords[:2] / pixel_coords[2]
    return x, y
def pixel_to_world_coordinates(x: float, y: float, K: torch.Tensor, pose: torch.Tensor, depth: float = 1.0):
    """
    Convert pixel coordinates to world coordinates.
    
    Args:
    x, y: Pixel coordinates
    K: Camera intrinsic matrix (3x3)
    pose: Camera extrinsic matrix (3x4 or 4x4)
    depth: Depth of the point in camera space (default is 1.0)
    
    Returns:
    world_point: 3D coordinates in world space
    """
    # Ensure K and pose are on the same device
    device = K.device
    pose = pose.to(device)

    # Convert pixel coordinates to homogeneous coordinates
    pixel = torch.tensor([x, y, 1.0], dtype=torch.float32, device=device)

    # Convert pixel coordinates to normalized camera coordinates
    normalized_cam = torch.linalg.inv(K) @ pixel

    # Scale the normalized coordinates based on the depth
    cam_coords = normalized_cam * depth

    # Ensure pose is 4x4
    if pose.shape == (3, 4):
        pose = torch.vstack((pose, torch.tensor([0, 0, 0, 1], device=device)))

    # Convert camera coordinates to world coordinates
    world_coords = pose @ torch.cat([cam_coords, torch.tensor([1.0], device=device)])

    return world_coords[:3]

def envmap2ballimg_pixel_space_2d(
    envmap: torch.Tensor,
    ball_params_2d: list,
    pose: torch.Tensor,
    K: torch.Tensor,
    image_height: int,
    image_width: int,
    interp_mode='bilinear'
):
    device, dtype = pose.device, pose.dtype
    x, y, r = ball_params_2d
    ball_center_2d = torch.tensor([x, y], device=device, dtype=dtype)

    # Create pixel coordinates
    pixel_xs = torch.arange(0, image_width, device=device, dtype=dtype)
    pixel_ys = torch.arange(0, image_height, device=device, dtype=dtype)
    pixel_xs, pixel_ys = torch.meshgrid(pixel_xs, pixel_ys, indexing='xy')
    pixels = torch.stack([pixel_xs, pixel_ys, torch.ones_like(pixel_xs)], dim=-1)

    # Calculate ball mask
    distances = torch.norm(pixels[..., :2] - ball_center_2d, dim=-1)
    ball_mask = distances <= r

    # Compute ball center in world space (assuming depth = 0)
    ball_center_homogeneous = torch.tensor([x, y, 1.0], device=device, dtype=dtype)
    ball_center_cam = torch.matmul(torch.inverse(K), ball_center_homogeneous)
    ball_center_cam = ball_center_cam / ball_center_cam[2]  # Normalize so z = 1
    ball_center_world = torch.matmul(pose[:3, :3], ball_center_cam) + pose[:3, 3]

    # Compute rays in world space
    K_inv = torch.inverse(K)
    rays_cam = torch.matmul(K_inv, pixels.reshape(-1, 3).T).T
    rays_world = torch.matmul(pose[:3, :3], rays_cam.T).T
    rays_world = rays_world / torch.norm(rays_world, dim=-1, keepdim=True)
    rays_world = rays_world.reshape(image_height, image_width, 3)

    # Camera position in world space
    cam_pos_world = pose[:3, 3]

    # Compute intersection with the ball
    oc = cam_pos_world - ball_center_world
    rays_masked = rays_world[ball_mask]
    oc_masked = oc.expand_as(rays_masked)
    
    a = torch.sum(rays_masked**2, dim=-1)
    b = 2 * torch.sum(oc_masked * rays_masked, dim=-1)
    c = torch.sum(oc_masked**2, dim=-1) - r**2
    
    discriminant = b**2 - 4*a*c
    valid_intersections = discriminant > 0
    
    t = (-b[valid_intersections] - torch.sqrt(discriminant[valid_intersections])) / (2*a[valid_intersections])
    
    # Compute intersection points
    intersections = cam_pos_world + t.unsqueeze(-1) * rays_masked[valid_intersections]

    # Compute surface normals
    normals = (intersections - ball_center_world) / r

    # Compute reflection vectors
    view_dirs = -rays_masked[valid_intersections]
    reflect_vecs = 2 * torch.sum(normals * view_dirs, dim=-1, keepdim=True) * normals - view_dirs

    # Convert reflection vectors to spherical coordinates
    thetas = torch.atan2(torch.sqrt(reflect_vecs[:, 0]**2 + reflect_vecs[:, 1]**2), reflect_vecs[:, 2])
    phis = torch.atan2(reflect_vecs[:, 1], reflect_vecs[:, 0])

    # Normalize spherical coordinates for grid_sample
    thetas = thetas / torch.pi - 0.5
    phis = -phis / (2 * torch.pi) + 0.5
    grid = torch.stack([phis, thetas], dim=1).view(1, -1, 1, 2)

    # Sample from the environment map
    sampled_colors = torch.nn.functional.grid_sample(
        envmap.unsqueeze(0),
        grid,
        mode=interp_mode,
        padding_mode='border',
        align_corners=True
    ).squeeze(0).squeeze(2).T

    # Create the output image
    output = torch.zeros((image_height, image_width, envmap.shape[0]), device=device, dtype=dtype)
    output[ball_mask] = sampled_colors

    return output.permute(2, 0, 1)  # [C, H, W]
def pixels_to_rays(pix_xs, pix_ys, K, pose):
    pixel_dirs = torch.stack([
        pix_xs + 0.5, pix_ys + 0.5, torch.ones_like(pix_xs)
    ], dim=-1)

    pix2cam = torch.linalg.inv(K)

    camera_dirs = torch.matmul(pix2cam, pixel_dirs[..., None])[..., 0]

    world_dirs = torch.matmul(pose[..., :3, :3], camera_dirs[..., None])[..., 0]
    world_dirs = world_dirs / torch.linalg.norm(world_dirs, dim=-1, keepdims=True)

    origins = torch.broadcast_to(pose[..., :3, -1], world_dirs.shape)
    return origins, world_dirs


def rotate_envmap(envmap, rotation_matrix, interp_mode='bilinear'):
    """Rotate the envmap by the rotation matrix.
        envmap: [B, C, H, W]
        rotation_matrix: [B, 3, 3]
    """
    rotation_matrix_inv = torch.linalg.inv(rotation_matrix.float()).to(envmap) # take a inverse since we do backward warping

    env_grid = create_envmap_grid(envmap.shape[-2]).to(envmap)
    thetas, phis = env_grid[...,0], env_grid[...,1]

    directions = get_cartesian_from_spherical(thetas, phis) # [H, W, 3]
    directions = torch.einsum('bij,hwj->bhwi', rotation_matrix_inv, directions)
    thetas, phis, rs = get_spherical_from_cartesian(directions[...,0], directions[...,1], directions[...,2], to_2pi=False)
    thetas = thetas / (torch.pi / 2) - 1 # [0, pi] -> [-1, 1]
    phis = -phis / torch.pi  # [pi, -pi] -> [-1, 1]
    grid = torch.stack([thetas, phis], dim=-1).expand(envmap.shape[0], -1, -1, -1) # [B, H, W, 2]

    new_envmap = torch.nn.functional.grid_sample(envmap, grid.flip(-1), mode=interp_mode, padding_mode='border', align_corners=True)
    return new_envmap


def get_ballimg_weights(ballimg):
    H, W = ballimg.shape[2:]
    i = torch.linspace(-1, 1, H).to(ballimg)
    j = torch.linspace(-1, 1, W).to(ballimg)
    ii, jj = torch.meshgrid(i, j, indexing='ij')
    dist = torch.sqrt(ii ** 2 + jj ** 2)
    weights = torch.clamp(1.0 - dist ** 2, 0.0, 1.0).unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
    return weights


def aggregate_multiview_ballimg_to_envmap(ballimgs, poses, envmap_size, method="weighted_avg"):
    envmap_canonicals = [
        ballimg2envmap(ballimgs[i].unsqueeze(0), envmap_size, pose=poses[i:i+1]) for i in range(len(ballimgs))
    ]
    envmap_canonicals = torch.concat(envmap_canonicals, dim=0)
    if method == "avg":
        envmap = torch.mean(envmap_canonicals, dim=0)
    elif method == "median":
        envmap = torch.median(envmap_canonicals, dim=0)[0]
    elif method == "weighted_avg":
        ballimg_weights = get_ballimg_weights(ballimgs[0].unsqueeze(0)).repeat(len(ballimgs), 1, 1, 1)
        envmap_weights = ballimg2envmap(ballimg_weights, envmap_size, pose=poses)
        envmap = torch.sum(envmap_canonicals * envmap_weights, dim=0) / torch.sum(envmap_weights, dim=0)
    elif method == "weighted_max":
        ballimg_weights = get_ballimg_weights(ballimgs[0].unsqueeze(0)).repeat(len(ballimgs), 1, 1, 1)
        envmap_weights = ballimg2envmap(ballimg_weights, envmap_size, pose=poses)
        max_idx = torch.argmax(envmap_weights, dim=0, keepdim=True)
        onehot = torch.zeros_like(envmap_weights)
        onehot.scatter_(0, max_idx, 1)
        # envmap = torch.gather(envmap_canonicals, 0, max_idx)[0]
        envmap = torch.sum(envmap_canonicals * onehot, dim=0)
    else:
        raise NotImplementedError(f"method {method} is not implemented")
    return envmap
def hdr2ldr_strict_positive(image, exposure, gamma=2.4, eps=1e-6):

   """
   改进的HDR到LDR转换
   """
   # 使用固定的参考值而不是每次都计算max
   reference = 1.0
   
   # 更好的负值处理方式
   image_positive = torch.where(image > 0, image, eps)
   
   # 曝光调整
   exposed = image_positive * (2.0 ** exposure)
   
   # 使用固定参考值进行归一化
   normalized = exposed / (reference + exposed)
   
   # gamma校正
   final = normalized ** (1/gamma)
   
   return torch.clamp(final, 0, 1)
def hdr2ldr(image, exposure, gamma=2.4):
    """
    参考论文方法，将 HDR 转换为 LDR:
    1. 先应用曝光调整
    2. 用 99% 分位数归一化，最高亮度映射到 0.9
    3. 进行 gamma 校正
    4. 进行裁剪，确保 LDR 范围 [0, 1]
    """
    # 曝光调整
    ldr = image * (2 ** exposure)

    # 计算 99% 分位数
    #percentile_99 = torch.quantile(ldr, 0.99)

    # 归一化到 0.9
    #ldr = ldr / (percentile_99 + 1e-6) * 0.90  
    #ldr = ldr/(1.0+ldr)

    # ReLU 处理负值
    ldr = torch.relu(ldr)
    #ldr = torch.clamp(ldr, 0, 1)

    # Gamma 校正
    
    #percentile_99 = torch.quantile(ldr, 0.50)
    #ldr = ldr / (percentile_99 + 1e-6) * 0.50
    #reinhard　tone mapping
    #ldr = ldr / (1 + ldr*0.9)
    
    ldr = ldr ** (1 / gamma)
    

    # 确保最终值在 [0,1] 范围内
    ldr = torch.clamp(ldr, 0, 1)

    return ldr

def aces_tone_mapping(image, exposure, gamma=2.4):
    # 应用曝光
    x = image * (2 ** exposure)
    x = torch.relu(x)
    
    # ACES色调映射
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    
    x = (x * (a * x + b)) / (x * (c * x + d) + e)
    
    # 伽马校正和裁剪
    x = x ** (1 / gamma)
    x = torch.clamp(x, 0, 1)
    
    return x
'''
def hdr2ldr(image, exposure, gamma=2.4):
    # image: range [0, +inf]
 
    def soft_clamp(x, min_val=0, max_val=1, margin=0.1):
        x = x + margin * torch.sigmoid((min_val - x) / margin)  # 在min_val附近平滑过渡
        x = x - margin * torch.sigmoid((x - max_val) / margin)  # 在max_val附近平滑过渡
        return x
    ldr = image * (2 ** exposure)

    ldr = torch.relu(ldr)+1e-4


    ldr = ldr ** (1 / gamma)
    #ldr = soft_clamp(ldr, min_val=0, max_val=1, margin=0.1)
    #hard clamp
    ldr = torch.clamp(ldr, 0, 1)+1e-4

    ldr = image * (2 ** exposure)
    ldr_mapped = (0.18 * ldr) / (1 + 0.18 * ldr)
    ldr_mapped = ldr_mapped ** (1 / gamma)
    ldr = ldr

    return ldr
'''
'''
def hdr2ldr(image, exposure, gamma=2.4):
    # 确保输入为正值
    eps = 1e-6
    image = torch.max(image, torch.tensor(eps))
    
    # 在log空间进行exposure调整
    # 避免直接乘以2^exposure可能导致的数值不稳定
    log_exposure = torch.log2(image) + exposure
    ldr = torch.exp2(log_exposure)
    
    # 平滑的Reinhard-style色调映射
    ldr = ldr / (2 + ldr)
    
    # gamma校正
    ldr = ldr ** (1/gamma)
    
    return ldr
'''
def ldr2hdr(image, exposure, gamma=2.4):
    # image: range [0, 1]
    hdr = image ** gamma
    hdr = hdr * (2 ** -exposure)
    return hdr


def rgb2luminance(image):
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    lumi = r * 0.212671 + g * 0.715160 + b * 0.072169
    return lumi


def get_saturated_mask(image, exposure, gamma=2.4, threshold=0.9):
    # image: ldr
    lumi = rgb2luminance(ldr2hdr(image, exposure, gamma))
    max_val = 2 ** -exposure
    mask = lumi > threshold * max_val
    return mask


def test_envmap_projection():
    def tensor2pil(tensor):
        arr = tensor[0].permute(1,2,0).numpy()
        arr = Image.fromarray((arr * 255).astype(np.uint8))
        return arr

    path = "example/abandoned_bakery_primary.png"
    envmap = Image.open(path).resize((512, 256))
    envmap = torch.tensor(np.array(envmap)).float().permute(2,0,1)[None] / 255.0

    envmap_canonical = envmap
    envmap_canonical_pil = tensor2pil(envmap_canonical)
    envmap_canonical_pil.save("bakery_envmap_canonical.png")

    ballimg_torch = envmap2ballimg(envmap, 256)
    ballimg = tensor2pil(ballimg_torch)
    ballimg.save("bakery_ball_canonical.png")

    envmap_canonical_rec = ballimg2envmap(ballimg_torch, 256)
    envmap_canonical_rec_pil = tensor2pil(envmap_canonical_rec)
    envmap_canonical_rec_pil.save("bakery_envmap_canonical_rec.png")

    # from -y look at origin
    R2 = torch.tensor(
        # [[1, 0, 0],
        #  [0, 0, -1],
        #  [0, 1, 0]]
        # [[0, 1, 0],
        #  [-1, 0, 0],
        #  [0, 0, 1]]
        ((-1.0, 0.0, 0.0, 0.0),
        (0.0, -0.8000000715255737, 0.6000000238418579, 6.0),
        (0.0, 0.5999999642372131, 0.800000011920929, 8.0),
        (0.0, 0.0, 0.0, 1.0))
        # (((1.0, 0.0, 0.0, 0.0),
        # (0.0, 1.0, 0.0, 0.0),
        # (0.0, 0.0, 1.0, 10.0),
        # (0.0, 0.0, 0.0, 1.0)))
    ).float()

    ballimg_r2_torch = envmap2ballimg(envmap, 256, R2[None])
    ballimg_r2 = tensor2pil(ballimg_r2_torch)
    ballimg_r2.save("bakery_ball_r2.png")
    
    # envmap_r2_rec = ballimg2envmap(ballimg_r2_torch, 256, R2[None])
    # envmap_r2_rec_pil = tensor2pil(envmap_r2_rec)
    # envmap_r2_rec_pil.save("bakery_envmap_r2_rec.png")


def test_ballimg_rec():
    from matplotlib import pyplot as plt

    def tensor2pil(tensor):
        arr = tensor[0, :3].permute(1,2,0).numpy()
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = Image.fromarray((arr * 255).astype(np.uint8))
        return arr
    
    pose = torch.tensor(
        ((-1.0, 0.0, 0.0, 0.0),
        (0.0, -0.8000000715255737, 0.6000000238418579, 6.0),
        (0.0, 0.5999999642372131, 0.800000011920929, 8.0),
        (0.0, 0.0, 0.0, 1.0))
    ).float()

    name = "12_zt"
    
    path = "outputs/multi_garden_4_first_zt_{ddim}_debug/debug/12_zt_before.npy"
    tensor = np.load(path)[0:1, :, 64:96, 64:96]
    ballimg_orig = torch.tensor(tensor).float()

    # print(tensor.shape)
    # exit()
    # path = "outputs/multi_garden_4_baseline_ddim/square/garden_x512_y512_r256_depth-nearest_ev-00_000.png"
    # ballimg_orig = Image.open(path).resize((32, 32))
    # ballimg_orig = torch.tensor(np.array(ballimg_orig)).float().permute(2,0,1)[None] / 255.0
    tensor2pil(ballimg_orig).save(f"tmp_out/{name}_orig.png")

    envmap = ballimg2envmap(ballimg_orig, 128, pose=pose[None], interp_mode='nearest')
    envmap_canonical_pil = tensor2pil(envmap)
    envmap_canonical_pil.save(f"tmp_out/{name}_env.png")

    ballimg_torch = envmap2ballimg(envmap, 32, pose=pose[None], interp_mode='nearest')
    ballimg_torch_mask = (ballimg_torch != 0).to(ballimg_torch)
    ballimg_torch = ballimg_torch * ballimg_torch_mask + ballimg_orig * (1 - ballimg_torch_mask)
    ballimg = tensor2pil(ballimg_torch)
    ballimg.save(f"tmp_out/{name}_rec.png")

    error = ballimg_orig - ballimg_torch
    error_rel = error.abs() / (ballimg_orig.abs())

    # error = (error - error.min()) / (error.max() - error.min())
    # tensor2pil(error).save(f"tmp_out/{name}_error.png")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].matshow(ballimg_orig[0, 1].numpy())
    axes[1].matshow(ballimg_torch[0, 1].numpy())
    axes[2].matshow(error[0, 1].numpy())
    plt.savefig(f"tmp_out/{name}_error.png")


if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    # test_envmap_projection()
    test_ballimg_rec()

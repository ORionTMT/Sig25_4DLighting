import math
from typing import Iterator
import torch
import torch.nn as nn
from abc import abstractmethod
import sys
import numpy as np
import imageio
from .envmap_utils import envmap2ballimg_perspective, create_envmap_grid, get_spherical_from_cartesian, canonical_camera_orientation, canonical_camera_zdir, envmap_grid_to_cartesian

def construct_envmap(args):
    if args.envmap_type == "grid3x3":
        envmap = EnvironmentMapGridStructure(
            height=args.envmap_height, width=args.envmap_width, init=args.envmap_init,
        )
    elif args.envmap_type == "grid3x3x3":
        envmap = EnvironmentMapGridThreeDim(
            height=args.envmap_height, width=args.envmap_width, grid_start = args.grid_start, grid_end = args.grid_end, depth=args.depth, init=args.envmap_init,
        )
    elif args.envmap_type == "grid":
        envmap = EnvironmentMapGrid(
            height=args.envmap_height, width=args.envmap_width, init=args.envmap_init,
        )
    elif args.envmap_type == "grid_linear":
        envmap = EnvironmentMapGridLinear(
            height=args.envmap_height, width=args.envmap_width
        )
    elif args.envmap_type == "grid_stack":
        envmap = EnvironmentMapGridStack(
            height=args.envmap_height, width=args.envmap_width
        )
    elif args.envmap_type == "mlp":
        envmap = EnvironmentMapMLP(
            height=args.envmap_height, width=args.envmap_width,
            hidden_dim=128, n_layers=3, posenc_deg=4
        )
    elif args.envmap_type == "mlp3d":
        envmap = EnvironmentMap3DMLP(
            height=args.envmap_height, width=args.envmap_width,
            hidden_dim=256, n_layers=6, posenc_deg=8, direnc_deg=8
        )
    elif args.envmap_type == "temporal":
        envmap = TemporalEnvironmentMap3DMLP(
            height=args.envmap_height, width=args.envmap_width,hidden_dim=256, n_layers=6, posenc_deg=6, direnc_deg=4, temporal_enc_deg=6, use_identity=False
        ) 
    elif args.envmap_type == "mlp3dsdxl":
        envmap = EnvironmentMap3DMLPSDXL(
            height=args.envmap_height, width=args.envmap_width,
            hidden_dim=64, n_layers=4, posenc_deg=6, direnc_deg=6
        )
    else:
        raise ValueError(f"Unknown envmap type: {args.envmap_type}")
    return envmap


class EnvironmentMapGridThreeDim:
    def __init__(self, height, width, grid_start, grid_end, depth, init="zero"):
        self.grid = [[[EnvironmentMapGrid(height, width, init) for _ in range(3)] for _ in range(3)] for _ in range(3)]
        self.height = height
        self.width = width
        self.max_depth = depth
        self.grid_start = grid_start
        self.grid_end = grid_end
        grid_middle = (grid_start + grid_end) / 2
        depth_min = 1
        depth_middle = ((depth-depth_min) / 2)+1
        self.grid_positions = torch.tensor([grid_start, grid_middle, grid_end])
        self.depth_positions = torch.tensor([1, depth_middle, depth])

    def get_envmap(self, row, col, depth):
        return self.grid[row][col][depth]

    def to_image(self, x, y, z, device, depth_ratio):
        # 1. 局部插值(使用最邻近的8个点)
        x_positions = self.grid_positions
        y_positions = self.grid_positions
        z_positions = self.depth_positions

        # 找到最近的网格索引
        x0_idx = torch.searchsorted(x_positions, x, right=True) - 1
        x0_idx = x0_idx.clamp(0, 1)
        x1_idx = x0_idx + 1
        x1_idx = x1_idx.clamp(1, 2)
        
        y0_idx = torch.searchsorted(y_positions, y, right=True) - 1
        y0_idx = y0_idx.clamp(0, 1)
        y1_idx = y0_idx + 1
        y1_idx = y1_idx.clamp(1, 2)
        
        z0_idx = torch.searchsorted(z_positions, z, right=True) - 1
        z0_idx = z0_idx.clamp(0, 1)
        z1_idx = z0_idx + 1
        z1_idx = z1_idx.clamp(1, 2)

        # 计算局部插值权重
        x0_pos = x_positions[x0_idx]
        x1_pos = x_positions[x1_idx]
        xd_local = (x - x0_pos) / (x1_pos - x0_pos + 1e-8)

        y0_pos = y_positions[y0_idx]
        y1_pos = y_positions[y1_idx]
        yd_local = (y - y0_pos) / (y1_pos - y0_pos + 1e-8)

        z0_pos = z_positions[z0_idx]
        z1_pos = z_positions[z1_idx]
        zd_local = (z - z0_pos) / (z1_pos - z0_pos + 1e-8)

        # 局部插值
        local_c000 = self.grid[y0_idx][x0_idx][z0_idx].forward()
        local_c001 = self.grid[y0_idx][x0_idx][z1_idx].forward()
        local_c010 = self.grid[y0_idx][x1_idx][z0_idx].forward()
        local_c011 = self.grid[y0_idx][x1_idx][z1_idx].forward()
        local_c100 = self.grid[y1_idx][x0_idx][z0_idx].forward()
        local_c101 = self.grid[y1_idx][x0_idx][z1_idx].forward()
        local_c110 = self.grid[y1_idx][x1_idx][z0_idx].forward()
        local_c111 = self.grid[y1_idx][x1_idx][z1_idx].forward()

        # 局部三线性插值
        local_c00 = local_c000 * (1 - zd_local) + local_c001 * zd_local
        local_c01 = local_c010 * (1 - zd_local) + local_c011 * zd_local
        local_c10 = local_c100 * (1 - zd_local) + local_c101 * zd_local
        local_c11 = local_c110 * (1 - zd_local) + local_c111 * zd_local

        local_c0 = local_c00 * (1 - yd_local) + local_c10 * yd_local
        local_c1 = local_c01 * (1 - yd_local) + local_c11 * yd_local

        local_interpolated = local_c0 * (1 - xd_local) + local_c1 * xd_local

        # 2. 全局插值(使用整个格子的8个角点)
        # 计算全局插值权重
        xd_global = (x - x_positions[0]) / (x_positions[-1] - x_positions[0] + 1e-8)
        yd_global = (y - y_positions[0]) / (y_positions[-1] - y_positions[0] + 1e-8)
        zd_global = (z - z_positions[0]) / (z_positions[-1] - z_positions[0] + 1e-8)

        # 全局插值使用最外层的8个点
        global_c000 = self.grid[0][0][0].forward()
        global_c001 = self.grid[0][0][2].forward()
        global_c010 = self.grid[0][2][0].forward()
        global_c011 = self.grid[0][2][2].forward()
        global_c100 = self.grid[2][0][0].forward()
        global_c101 = self.grid[2][0][2].forward()
        global_c110 = self.grid[2][2][0].forward()
        global_c111 = self.grid[2][2][2].forward()

        # 全局三线性插值
        global_c00 = global_c000 * (1 - zd_global) + global_c001 * zd_global
        global_c01 = global_c010 * (1 - zd_global) + global_c011 * zd_global
        global_c10 = global_c100 * (1 - zd_global) + global_c101 * zd_global
        global_c11 = global_c110 * (1 - zd_global) + global_c111 * zd_global

        global_c0 = global_c00 * (1 - yd_global) + global_c10 * yd_global
        global_c1 = global_c01 * (1 - yd_global) + global_c11 * yd_global

        global_interpolated = global_c0 * (1 - xd_global) + global_c1 * xd_global

        # 3. 计算局部和全局的混合权重
        # 距离中心的归一化距离作为混合权重
        center_x = (x_positions[0] + x_positions[-1]) / 2
        center_y = (y_positions[0] + y_positions[-1]) / 2
        center_z = (z_positions[0] + z_positions[-1]) / 2
        
        dist_to_center = torch.sqrt(
            ((x - center_x)/(x_positions[-1] - x_positions[0]))**2 + 
            ((y - center_y)/(y_positions[-1] - y_positions[0]))**2 + 
            ((z - center_z)/(z_positions[-1] - z_positions[0]))**2
        )
        
        # 使用sigmoid函数使过渡更平滑
        #alpha = torch.sigmoid(5 * (dist_to_center - 0.5))  # 可以调整系数5和偏移0.5来控制过渡的陡峭程度
        alpha = 0.9
        
        # 混合局部和全局插值结果
        #print("alpha", alpha)
        interpolated = local_interpolated
        
        return interpolated


    def find_bounds(self, coord, coord_list):
        lower = max([c for c in coord_list if c <= coord], default=min(coord_list))
        upper = min([c for c in coord_list if c >= coord], default=max(coord_list))
        return lower, upper

    def query_by_position(self, x, y, z):
        return self.interpolate(x, y, z)

    def to(self, device):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    self.grid[i][j][k] = self.grid[i][j][k].to(device)
        return self
    def parameters(self):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    yield from self.grid[i][j][k].parameters()
    def save_as_exr(self, envmap_tensor, file_path):
        
        envmap_np = envmap_tensor.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)  
        
        # Save EXR HDR image
        imageio.imwrite(file_path, envmap_np, format='EXR')

        print(f"Saved environment map as {file_path}")

class EnvironmentMapGridStructure:
    def __init__(self, height, width, init="zero"):
        self.grid = [[EnvironmentMapGrid(height, width, init) for _ in range(3)] for _ in range(3)]
        self.height = height
        self.width = width
        self.grid_positions = [100, 512,924]


    def get_envmap(self, row, col):
        return self.grid[row][col]

    
    def interpolate(self, x, y):
        x0, x1 = self.find_bounds(x, self.grid_positions)
        y0, y1 = self.find_bounds(y, self.grid_positions)

        # Calculate the grid indices
        i0 = min(max(self.grid_positions.index(x0), 0), 2)
        i1 = min(max(i0 + 1, 0), 2)
        j0 = min(max(self.grid_positions.index(y0), 0), 2)
        j1 = min(max(j0 + 1, 0), 2)

        # Get the four surrounding EnvMaps
        f00 = self.grid[j0][i0].forward()
        f01 = self.grid[j0][i1].forward()
        f10 = self.grid[j1][i0].forward()
        f11 = self.grid[j1][i1].forward()

        # Calculate interpolation weights
        wx = (x - x0) / (x1 - x0) if x1 != x0 else 0
        wy = (y - y0) / (y1 - y0) if y1 != y0 else 0

        # Perform bilinear interpolation
        interpolated = (
            f00 * (1 - wx) * (1 - wy) +
            f01 * wx * (1 - wy) +
            f10 * (1 - wx) * wy +
            f11 * wx * wy
        )

        return interpolated

    def find_bounds(self, coord, coord_list):
        lower = max([c for c in coord_list if c <= coord], default=min(coord_list))
        upper = min([c for c in coord_list if c >= coord], default=max(coord_list))
        return lower, upper

    def query_by_position(self, x, y):
        # Assuming x and y are normalized coordinates in [0, 1]
        row = y * 2
        col = x * 2
        return self.interpolate(row, col)
    def to(self, device):
        for row in self.grid:
            for envmap in row:
                envmap.to(device)
        return self
    def parameters(self):
        for i in range(3):
            for j in range(3):
                    yield from self.grid[i][j].parameters()
    def save_as_exr(self, envmap_tensor, file_path):
            
            envmap_np = envmap_tensor.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)  
            
            # Save EXR HDR image
            imageio.imwrite(file_path, envmap_np, format='EXR')
    
            print(f"Saved environment map as {file_path}")
    

class EnvironmentMap(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def trainable_parameters(self, ev):
        return self.parameters()

    def project_to_ball(self, pose, K, ball_param, image_width=None, image_height=None, **kwargs):
        return None
    
    def to_image(self, ev):
        return None
    
    def forward(self):
        return None



class EnvironmentMapGrid(EnvironmentMap):
    def __init__(self, height, width, init="zero") -> None:
        super().__init__()
        if init == "zero":
            #grid = torch.zeros(1, 3, height, width)+0.2
            # 小方差高斯分布,避免初始值过大
           grid = torch.zeros(1, 3, height, width)
        elif init == "normal":
            grid = torch.randn(1, 3, height, width) 
        elif init == "black":
            grid = torch.zeros(1, 3, height, width) - 2
        else:
            raise NotImplementedError(f"init {init} is not implemented")
        self.grid = nn.Parameter(grid)

        self.register_buffer(
            "mask", torch.ones((height, width))
        )

    def set_mask(self, mask):
        self.mask = mask

    def project_to_ball(self, pose, K, ball_param, image_width=None, image_height=None, interp_mode='bilinear', ev=None):
        envmap = self.forward() # [-1, 1]
        ball_img = envmap2ballimg_perspective(
            envmap[0], ball_param, pose, K,
            image_width=image_width, image_height=image_height, interp_mode=interp_mode
        )
        return ball_img
    
    def to_raw(self):
        return self.forward()
    
    def to_image(self, ev):
        return self.forward().clamp(-1, 1)

    def forward(self):
        # d: direction [B, 3]
        #envmap = self.grid.tanh()
        #use a relu activation
        #softplus = torch.nn.Softplus()
        envmap = torch.nn.functional.softplus(self.grid)+1e-6
        envmap = envmap * self.mask + envmap.detach() * (1 - self.mask) # only keep gradient for masked part
        return envmap


class EnvironmentMapGridLinear(EnvironmentMap):
    def __init__(self, height, width, init="zero", raw_func="sigmoid") -> None:
        super().__init__()
        if init == "zero": # init to be 0.5
            grid = torch.zeros(1, 3, height, width)
            if raw_func == "sigmoid":
                grid = grid - 4 # init to be sigmoid(-4)
            elif raw_func == "linear":
                grid = grid + 0.5 
            elif raw_func == "exp":
                grid = grid + math.log(0.5)
        elif init == "normal":
            grid = torch.randn(1, 3, height, width) * 0.2
        else:
            raise NotImplementedError(f"init {init} is not implemented")
        self.grid = nn.Parameter(grid)
        self.raw_func = raw_func

    def project_to_ball(self, pose, K, ball_param, image_width=None, image_height=None, interp_mode='bilinear', ev=0):
        envmap = self.forward(ev) # [-1, 1]
        ball_img = envmap2ballimg_perspective(
            envmap[0], ball_param, pose, K,
            image_width=image_width, image_height=image_height, interp_mode=interp_mode
        )
        return ball_img
    
    def to_raw(self):
        if self.raw_func == "sigmoid":
            envmap = torch.sigmoid(self.grid) * 2 ** 5
        elif self.raw_func == "linear":
            envmap = torch.nn.functional.softplus(self.grid, beta=10)
        elif self.raw_func == "exp":
            envmap = torch.exp(self.grid)
        else:
            raise NotImplementedError
        return envmap
    
    def to_image(self, ev):
        return self.forward(ev)

    def clamp01(self, x):
        # piece-wise linear clamp
        return torch.maximum(torch.minimum(x, x * 0.0001 + 0.9999), x * 0.0001)
    
    def forward(self, exposure, gamma=2.4):
        envmap = self.to_raw() # [0, +inf]

        envmap = envmap * (2 ** exposure)
        envmap = envmap ** (1. / gamma) # [0, +inf]
        envmap = self.clamp01(envmap) # [0-eps, 1+eps]
        envmap = envmap * 2 - 1 # [-1, 1]
        return envmap


class EnvironmentMapGridMultiExposure(EnvironmentMap):
    def __init__(self, height, width, init="zero", ev_list=[0, -2.5, -5]) -> None:
        super().__init__()
        if init == "zero":
            grid = torch.zeros(1, 3, height, width) + 1e-6
        elif init == "normal":
            grid = torch.randn(1, 3, height, width) * 0.2
        else:
            raise NotImplementedError(f"init {init} is not implemented")
        # grid[0] = 0.5
        self.register_parameter("grid_0", nn.Parameter(grid))
        self.register_parameter("grid_1", nn.Parameter(torch.zeros_like(grid) - 4))
        self.register_parameter("grid_2", nn.Parameter(torch.zeros_like(grid) - 4))
        self.exposures = ev_list
        self.height = height
        self.width = width
        # self.register_buffer("exposure", tensor=torch.tensor(ev_list).reshape(-1, 1, 1, 1))
        # self.register_buffer(
        #     "mask_0", torch.ones(1, 1, height, width)
        # )

    def trainable_parameters(self, ev):
        return self.parameters()

    def project_to_ball(self, pose, K, ball_param, image_width=None, image_height=None, interp_mode='bilinear', ev=0):
        envmap = self.forward(ev) # [-1, 1]
        ball_img = envmap2ballimg_perspective(
            envmap[0], ball_param, pose, K,
            image_width=image_width, image_height=image_height, interp_mode=interp_mode
        )
        return ball_img
    
    def clamp01(self, x):
        # piece-wise linear clamp
        return torch.maximum(torch.minimum(x, x * 0.0001 + 0.9999), x * 0.0001)
    
    def to_raw(self):
        n_ev = len(self.exposures)
        grid_evs = [getattr(self, f"grid_{i}") for i in range(n_ev)]
        # mask_evs = [getattr(self, f"mask_{i}") for i in range(n_ev)]
        ev_high_weight = [2 ** -self.exposures[i] for i in range(n_ev)]
        ev_low_weight = [0] + ev_high_weight[:-1]
        
        # envmap = sum([self.clamp01(g) * m * (h - l) for g, m, h, l in zip(grid_evs, mask_evs, ev_high_weight, ev_low_weight)])
        envmap = sum([torch.sigmoid(g) * (h - l) for g, h, l in zip(grid_evs, ev_high_weight, ev_low_weight)])
        return envmap

    def to_image(self, ev):
        return self.forward(ev).clamp(-1, 1)

    def forward(self, exposure, gamma=2.4):
        envmap = self.to_raw() # [0, +inf]
    
        envmap = envmap * (2 ** exposure)
        envmap = envmap ** (1. / gamma) # [0, +inf]
        envmap = self.clamp01(envmap) # [0-eps, 1+eps]
        envmap = envmap * 2 - 1 # [-1, 1]
        return envmap


class EnvironmentMapGridStack(EnvironmentMap):
    def __init__(self, height, width, init="zero", ev_start=0.0) -> None:
        super().__init__()
        if init == "zero":
            grid = torch.zeros(1, 3, height, width) + 1e-6
        elif init == "normal":
            grid = torch.randn(1, 3, height, width) * 0.2
        else:
            raise NotImplementedError(f"init {init} is not implemented")
        grid[0] = 0.5
        self.register_parameter("grid_0", nn.Parameter(grid))
        self.exposures = [ev_start]
        self.height = height
        self.width = width
        # self.register_buffer("exposure", tensor=torch.tensor(ev_list).reshape(-1, 1, 1, 1))

    def add_level(self, ev):
        self.exposures.append(ev)
        grid = torch.zeros(1, 3, self.height, self.width).to(getattr(self, "grid_0"))
        self.register_parameter(f"grid_{len(self.exposures) - 1}", nn.Parameter(grid))

    def trainable_parameters(self, ev):
        idx = self.exposures.index(ev)
        return [getattr(self, f"grid_{idx}")]
        # return [self.grid_0]

    def project_to_ball(self, pose, K, ball_param, image_width=None, image_height=None, interp_mode='bilinear', ev=0):
        envmap = self.forward(ev) # [-1, 1]
        ball_img = envmap2ballimg_perspective(
            envmap[0], ball_param, pose, K,
            image_width=image_width, image_height=image_height, interp_mode=interp_mode
        )
        return ball_img
    
    def clamp01(self, x):
        # piece-wise linear clamp
        return torch.maximum(torch.minimum(x, x * 0.0001 + 0.9999), x * 0.0001)
    
    def to_raw(self):
        n_ev = len(self.exposures)
        grid_evs = [getattr(self, f"grid_{i}") for i in range(n_ev)]
        ev_high_weight = [2 ** -self.exposures[i] for i in range(n_ev)]
        ev_low_weight = [0] + ev_high_weight[:-1]
        
        envmap = sum([self.clamp01(g) * (h - l) for g, h, l in zip(grid_evs, ev_high_weight, ev_low_weight)])
        return envmap

    def to_image(self, ev):
        return self.forward(ev).clamp(-1, 1)

    def forward(self, exposure, gamma=2.4):
        envmap = self.to_raw() # [0, +inf]
    
        envmap = envmap * (2 ** exposure)
        # envmap = envmap ** (1. / gamma) # [0, +inf]
        envmap = self.clamp01(envmap) # [0-eps, 1+eps]
        envmap = envmap * 2 - 1 # [-1, 1]
        return envmap


class SinusoidalEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, use_identity: bool = True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.register_buffer(
            "scales", torch.tensor([2**i for i in range(min_deg, max_deg)])
        )

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + (self.max_deg - self.min_deg) * 2
        ) * self.x_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[Ellipsis, None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim],
        )
        latent = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent
    

class EnvironmentMapMLP(EnvironmentMap):
    def __init__(self, height, width, hidden_dim, n_layers, posenc_deg=4):
        super().__init__()
        self.posenc = SinusoidalEncoder(2, 0, posenc_deg, use_identity=False)

        in_dim = self.posenc.latent_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
            ) for _ in range(n_layers - 1)],
            nn.Linear(hidden_dim, 3)
        )

        self.height = height
        self.width = width

    def project_to_ball(self, pose, K, ball_param, image_width=None, image_height=None, interp_mode='bilinear', ev=0):
        raise NotImplementedError
        # pose: [B, 3, 4]
        # return: [B, 3, H, W]
        orientation = pose[:, :3, :3]
        # apply world-to-cam1, cam2-to-world
        rotation = (
            orientation.float() @ torch.linalg.inv(canonical_camera_orientation.to(pose).float())
        ).to(pose)

        # canonical space rays
        z = torch.linspace(1, -1, ball_size).to(pose)
        y = torch.linspace(1, -1, ball_size).to(pose)
        z, y = torch.meshgrid(z, y, indexing='ij')
        x2 = (1 - y**2 - z**2)
        mask = x2 >= 0 # [H, W]
        x = -torch.clip(x2, 0, 1).sqrt()

        normal = torch.stack([x, y, z], dim=-1) # [H, W, 3]
        normal = normal / torch.linalg.norm(normal, dim=-1, keepdims=True)

        I = canonical_camera_zdir.to(pose)
        reflect_vec = 2 * torch.sum(normal * I, dim=-1, keepdims=True) * normal - I

        # transform
        reflect_vec_world = torch.einsum('bij,hwj->bhwi', rotation, reflect_vec)
        
        thetas, phis, rs = get_spherical_from_cartesian(
            reflect_vec_world[...,0], reflect_vec_world[...,1], reflect_vec_world[...,2], to_2pi=False
        )
        theta_phi_grid = torch.stack([thetas, phis], dim=-1)
        ball_img = self.forward(theta_phi_grid).permute(0, 3, 1, 2) # [B, 3, H, W]
        ball_img = ball_img * mask.unsqueeze(0).unsqueeze(1)

        return ball_img

    def to_image(self, size=256):
        size = size or self.height
        dtype = self.mlp[0].weight.dtype
        device = self.mlp[0].weight.device
        theta_phi_grid = create_envmap_grid(size).to(dtype=dtype, device=device)
        envmap = self.forward(theta_phi_grid).unsqueeze(0).permute(0, 3, 1, 2) # [1, 3, H, W]
        return envmap
    
    def forward(self, x):
        # x: [B, 2], theta, phi
        x = self.posenc(x)
        x = self.mlp(x)
        x = torch.tanh(x) # [-1, 1]
        return x
class HDROutput(nn.Module):
    def __init__(self, min_value=0.1, scale=10.0):
        super().__init__()
        self.min_value = min_value
        self.scale = scale
    
    def forward(self, x):
        # 确保有最小亮度
        base = self.min_value + self.scale * torch.relu(x)
        # 对每个通道独立处理
        return base

class EnvironmentMap3DMLP(nn.Module):
    def __init__(self, height, width, hidden_dim, n_layers, posenc_deg=2, direnc_deg=2, use_identity=False):
        super().__init__()

        self.pos_encoder = SinusoidalEncoder(3, 0, posenc_deg, use_identity=True)
        self.dir_encoder = SinusoidalEncoder(3, 0, direnc_deg, use_identity=True)
        
        self.grids = create_envmap_grid(height)
        self.cartesian = envmap_grid_to_cartesian(self.grids)

        pos_dim = self.pos_encoder.latent_dim
        in_dim = pos_dim + self.dir_encoder.latent_dim

        # Define the number of layers and the index of the middle layer
        num_layers = n_layers + 1  # Total number of linear layers
        middle_layer_index = num_layers // 2

        # Initial layers before the skip connection
        self.initial_layer = nn.Linear(in_dim, hidden_dim)
        #self.initial_activation = nn.SiLU()
        self.initial_activation = nn.LeakyReLU()

        self.pre_skip_layers = nn.ModuleList()
        for _ in range(middle_layer_index - 1):
            self.pre_skip_layers.append(nn.Linear(hidden_dim, hidden_dim))
            #self.pre_skip_layers.append(nn.SiLU())
            self.pre_skip_layers.append(nn.LeakyReLU())

        # Skip connection layer
        self.skip_layer = nn.Linear(hidden_dim + in_dim, hidden_dim)
        #self.skip_activation = nn.SiLU()
        self.skip_activation = nn.LeakyReLU()

        # Layers after the skip connection
        self.post_skip_layers = nn.ModuleList()
        for _ in range(num_layers - middle_layer_index - 1):
            self.post_skip_layers.append(nn.Linear(hidden_dim, hidden_dim))
            #self.post_skip_layers.append(nn.SiLU())
            self.post_skip_layers.append(nn.LeakyReLU())

        # Directional layers remain the same
        self.dir_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            #nn.SiLU(),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 3),
            #nn.Softplus()
            nn.Softplus()
        )

        self.height = height
        self.width = width

    def forward(self, pos, dirs, device):
        pos_enc = self.pos_encoder(pos)
        dirs_enc = self.dir_encoder(dirs)
        pos_enc = pos_enc.repeat(dirs_enc.shape[0], dirs_enc.shape[1], 1)

        inputs = torch.cat([pos_enc, dirs_enc], dim=-1)

        x = self.initial_layer(inputs)
        x = self.initial_activation(x)

        # Save the original encoded input for the skip connection
        original_encoded_input = inputs

        # Pass through layers before the skip connection
        for layer in self.pre_skip_layers:
            x = layer(x)

        # Skip connection layer
        x = torch.cat([x, original_encoded_input], dim=-1)
        x = self.skip_layer(x)
        x = self.skip_activation(x)

        # Pass through layers after the skip connection
        for layer in self.post_skip_layers:
            x = layer(x)

        # Directional layers
        rgb = self.dir_layers(x)
        #squared
        #rgb = torch.exp(rgb)
        rgb = rgb

        return rgb  # [1, 3, H, W]
        
    def to_image(self, x, y, z, device, depth_ratio):
        x = 2 * (x - 55) / (512 - 2 * 55) - 1
        y = 2 * (y - 55) / (512 - 2 * 55) - 1
        z = 2 * (math.log((z)) / math.log((512 / depth_ratio)+1)) - 1
        #if z>0:
            #print("z is positive", z)
        device = self.initial_layer.weight.device
        pos = torch.tensor([x, y, z]).to(device, dtype=torch.float32)
        dtype = self.initial_layer.weight.dtype
        

        cartesian = self.cartesian.to(dtype=dtype, device=device)

        envmap = self.forward(pos, cartesian, device).unsqueeze(0).permute(0, 3, 1, 2)  # [1, 3, H, W]
        return envmap
    
    def save_as_exr(self, envmap_tensor, file_path):
        envmap_np = envmap_tensor.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)  
        imageio.imwrite(file_path, envmap_np, format='EXR')
        print(f"Saved environment map as {file_path}")
class EnvironmentMap3DMLPSDXL(nn.Module):
    def __init__(self, height, width, hidden_dim, n_layers, posenc_deg=2, direnc_deg=2, use_identity=False):
        super().__init__()

        self.pos_encoder = SinusoidalEncoder(3, 0, posenc_deg, use_identity=True)
        self.dir_encoder = SinusoidalEncoder(3, 0, direnc_deg, use_identity=True)
        
        self.grids = create_envmap_grid(height)
        self.cartesian = envmap_grid_to_cartesian(self.grids)

        pos_dim = self.pos_encoder.latent_dim
        in_dim = pos_dim + self.dir_encoder.latent_dim

        # Define the number of layers and the index of the middle layer
        num_layers = n_layers + 1  # Total number of linear layers
        middle_layer_index = num_layers // 2

        # Initial layers before the skip connection
        self.initial_layer = nn.Linear(in_dim, hidden_dim)
        self.initial_activation = nn.LeakyReLU()

        self.pre_skip_layers = nn.ModuleList()
        for _ in range(middle_layer_index - 1):
            self.pre_skip_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.pre_skip_layers.append(nn.LeakyReLU())

        # Skip connection layer
        self.skip_layer = nn.Linear(hidden_dim + in_dim, hidden_dim)
        self.skip_activation = nn.LeakyReLU()

        # Layers after the skip connection
        self.post_skip_layers = nn.ModuleList()
        for _ in range(num_layers - middle_layer_index - 1):
            self.post_skip_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.post_skip_layers.append(nn.LeakyReLU())

        # Directional layers remain the same
        self.dir_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Softplus()
        )

        self.height = height
        self.width = width

    def forward(self, pos, dirs, device):
        pos_enc = self.pos_encoder(pos)
        dirs_enc = self.dir_encoder(dirs)
        pos_enc = pos_enc.repeat(dirs_enc.shape[0], dirs_enc.shape[1], 1)

        inputs = torch.cat([pos_enc, dirs_enc], dim=-1)

        x = self.initial_layer(inputs)
        x = self.initial_activation(x)

        # Save the original encoded input for the skip connection
        original_encoded_input = inputs

        # Pass through layers before the skip connection
        for layer in self.pre_skip_layers:
            x = layer(x)

        # Skip connection layer
        x = torch.cat([x, original_encoded_input], dim=-1)
        x = self.skip_layer(x)
        x = self.skip_activation(x)

        # Pass through layers after the skip connection
        for layer in self.post_skip_layers:
            x = layer(x)

        # Directional layers
        rgb = self.dir_layers(x)
        rgb = rgb 

        return rgb  # [1, 3, H, W]
        
    def to_image(self, x, y, z, device, depth_ratio):
        # 调整边界padding从55到110，以适应1024分辨率
        x = 2 * (x - 110) / (1024 - 2 * 110) - 1
        y = 2 * (y - 110) / (1024 - 2 * 110) - 1
        z = 2 * (math.log((z)) / math.log((1024 / depth_ratio)+1)) - 1

        device = self.initial_layer.weight.device
        pos = torch.tensor([x, y, z]).to(device, dtype=torch.float32)
        dtype = self.initial_layer.weight.dtype
        
        cartesian = self.cartesian.to(dtype=dtype, device=device)

        envmap = self.forward(pos, cartesian, device).unsqueeze(0).permute(0, 3, 1, 2)  # [1, 3, H, W]
        return envmap
    
    def save_as_exr(self, envmap_tensor, file_path):
        envmap_np = envmap_tensor.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)  
        imageio.imwrite(file_path, envmap_np, format='EXR')
        print(f"Saved environment map as {file_path}")
def positional_encoding(input, L):
    # input: (N, 3)
    encodings = [input]
    for i in range(L):
        for fn in [torch.sin, torch.cos]:
            encodings.append(fn(2.0 ** i * input))
    return torch.cat(encodings, dim=-1)

class EnvmapMLPSimplified(nn.Module):
    def __init__(self, input_dim):
        super(EnvmapMLPSimplified, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Sigmoid() 
        )
    
    def forward(self, x):
        return self.layers(x)
    

class TemporalEnvironmentMap3DMLP(nn.Module):
    def __init__(self, height, width, hidden_dim, n_layers, posenc_deg=2, direnc_deg=2, temporal_enc_deg=2, use_identity=False):
        super().__init__()

        # Add temporal encoder for time dimension
        self.pos_encoder = SinusoidalEncoder(3, 0, posenc_deg, use_identity=True)
        self.dir_encoder = SinusoidalEncoder(3, 0, direnc_deg, use_identity=True)
        self.temporal_encoder = SinusoidalEncoder(1, 0, temporal_enc_deg, use_identity=True)  # New encoder for time
        
        self.grids = create_envmap_grid(height)
        self.cartesian = envmap_grid_to_cartesian(self.grids)

        pos_dim = self.pos_encoder.latent_dim
        temporal_dim = self.temporal_encoder.latent_dim  # Dimension after temporal encoding
        in_dim = pos_dim + self.dir_encoder.latent_dim + temporal_dim  # Add temporal dimension

        # Rest of the network structure remains similar
        num_layers = n_layers + 1
        middle_layer_index = num_layers // 2

        self.initial_layer = nn.Linear(in_dim, hidden_dim)
        self.initial_activation = nn.LeakyReLU()

        self.pre_skip_layers = nn.ModuleList()
        for _ in range(middle_layer_index - 1):
            self.pre_skip_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.pre_skip_layers.append(nn.LeakyReLU())

        self.skip_layer = nn.Linear(hidden_dim + in_dim, hidden_dim)
        self.skip_activation = nn.LeakyReLU()

        self.post_skip_layers = nn.ModuleList()
        for _ in range(num_layers - middle_layer_index - 1):
            self.post_skip_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.post_skip_layers.append(nn.LeakyReLU())

        self.dir_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Softplus()
        )

        self.height = height
        self.width = width

    def forward(self, pos, dirs, t, device):
        # Encode position, direction and time
        pos_enc = self.pos_encoder(pos)
        dirs_enc = self.dir_encoder(dirs)
        t_enc = self.temporal_encoder(t.unsqueeze(-1))  # Encode time dimension
        
        # Repeat position and time encodings to match direction encoding dimensions
        pos_enc = pos_enc.repeat(dirs_enc.shape[0], dirs_enc.shape[1], 1)
        t_enc = t_enc.repeat(dirs_enc.shape[0], dirs_enc.shape[1], 1)

        # Concatenate all encoded inputs
        inputs = torch.cat([pos_enc, dirs_enc, t_enc], dim=-1)

        x = self.initial_layer(inputs)
        x = self.initial_activation(x)

        original_encoded_input = inputs

        for layer in self.pre_skip_layers:
            x = layer(x)

        x = torch.cat([x, original_encoded_input], dim=-1)
        x = self.skip_layer(x)
        x = self.skip_activation(x)

        for layer in self.post_skip_layers:
            x = layer(x)

        rgb = self.dir_layers(x)
        return rgb

    def to_image(self, x, y, z, t, device, depth_ratio):
        # Normalize spatial coordinates
        x = 2 * (x - 55) / (512 - 2 * 55) - 1
        y = 2 * (y - 55) / (512 - 2 * 55) - 1
        z = 2 * (math.log((z)) / math.log((512 / depth_ratio)+1)) - 1
        
        # Normalize temporal coordinate (assuming t is in range [1, max_frames])
        t_normalized = 2 * (t - 1) / (30 - 1) - 1  # For 5 frames example
        
        device = self.initial_layer.weight.device
        pos = torch.tensor([x, y, z]).to(device, dtype=torch.float32)
        t_tensor = torch.tensor(t_normalized).to(device, dtype=torch.float32)
        dtype = self.initial_layer.weight.dtype

        cartesian = self.cartesian.to(dtype=dtype, device=device)

        envmap = self.forward(pos, cartesian, t_tensor, device).unsqueeze(0).permute(0, 3, 1, 2)
        return envmap
    
    def save_exr(self, envmap_tensor, file_path):
        envmap_np = envmap_tensor.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)  
        imageio.imwrite(file_path, envmap_np, format='EXR')
        print(f"Saved environment map as {file_path}")


class MultiscaleLightField3DMLP(nn.Module):
    def __init__(self, height, width, hidden_dim, n_layers, num_scales=3, 
                 posenc_deg=2, direnc_deg=2, temporal_enc_deg=2, levelenc_deg=1, use_identity=False):
        super().__init__()

        # Encoders for different dimensions
        self.pos_encoder = SinusoidalEncoder(3, 0, posenc_deg, use_identity=True)
        self.dir_encoder = SinusoidalEncoder(3, 0, direnc_deg, use_identity=True)
        self.temporal_encoder = SinusoidalEncoder(1, 0, temporal_enc_deg, use_identity=True)
        self.level_encoder = SinusoidalEncoder(1, 0, levelenc_deg, use_identity=True)  # New encoder for scale level
        
        self.num_scales = num_scales
        self.grids = create_envmap_grid(height)
        self.cartesian = envmap_grid_to_cartesian(self.grids)

        # Calculate input dimensions
        pos_dim = self.pos_encoder.latent_dim
        dir_dim = self.dir_encoder.latent_dim
        temporal_dim = self.temporal_encoder.latent_dim
        level_dim = self.level_encoder.latent_dim
        
        in_dim = pos_dim + dir_dim + temporal_dim + level_dim

        # Network architecture
        num_layers = n_layers + 1
        middle_layer_index = num_layers // 2

        self.initial_layer = nn.Linear(in_dim, hidden_dim)
        self.initial_activation = nn.LeakyReLU()

        self.pre_skip_layers = nn.ModuleList()
        for _ in range(middle_layer_index - 1):
            self.pre_skip_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.pre_skip_layers.append(nn.LeakyReLU())

        self.skip_layer = nn.Linear(hidden_dim + in_dim, hidden_dim)
        self.skip_activation = nn.LeakyReLU()

        self.post_skip_layers = nn.ModuleList()
        for _ in range(num_layers - middle_layer_index - 1):
            self.post_skip_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.post_skip_layers.append(nn.LeakyReLU())

        # Output layers - one for each scale level for better multi-scale representation
        self.dir_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim // 2, 4),  # RGBA output (3 for RGB + 1 for alpha)
                nn.Softplus()
            ) for _ in range(num_scales)
        ])

        self.height = height
        self.width = width
        
        # Define scale-specific parameters
        # Each scale covers a certain spatial range
        self.scale_ranges = []
        for s in range(num_scales):
            # Scale 0 is the largest (coarsest), each subsequent scale is half the size
            scale_factor = 0.5 ** s
            self.scale_ranges.append(scale_factor)

    def forward(self, pos, dirs, t, level, device):
        """
        Forward pass through the network
        Args:
            pos: [B, 3] position coordinates
            dirs: [B, N, 3] direction vectors
            t: [B] time values
            level: [B, 1] or float, scale level (0 to num_scales-1)
            device: torch device
        """
        # Encode all inputs
        pos_enc = self.pos_encoder(pos)
        dirs_enc = self.dir_encoder(dirs)
        t_enc = self.temporal_encoder(t.unsqueeze(-1))
        
        # Handle level input
        if isinstance(level, float) or isinstance(level, int):
            level = torch.tensor([[level]], device=device, dtype=torch.float32).expand(pos.shape[0], 1)
        level_enc = self.level_encoder(level)
        
        # Repeat encodings to match direction encoding dimensions
        pos_enc = pos_enc.repeat(1, dirs_enc.shape[1], 1)
        t_enc = t_enc.repeat(1, dirs_enc.shape[1], 1)
        level_enc = level_enc.repeat(1, dirs_enc.shape[1], 1)

        # Concatenate all encoded inputs
        inputs = torch.cat([pos_enc, dirs_enc, t_enc, level_enc], dim=-1)

        # Forward pass through shared network
        x = self.initial_layer(inputs)
        x = self.initial_activation(x)

        original_encoded_input = inputs

        for layer in self.pre_skip_layers:
            x = layer(x)

        x = torch.cat([x, original_encoded_input], dim=-1)
        x = self.skip_layer(x)
        x = self.skip_activation(x)

        for layer in self.post_skip_layers:
            x = layer(x)
            
        # Compute continuous level weights
        if isinstance(level, torch.Tensor) and level.shape[-1] == 1:
            # Convert level to value between 0 and num_scales-1
            level_val = level.squeeze(-1)
            
            # Get integer and fractional parts
            level_int = torch.floor(level_val).long()
            level_frac = level_val - level_int
            
            # Ensure we stay within bounds
            level_int = torch.clamp(level_int, 0, self.num_scales - 2)
            
            # Get outputs from adjacent scales and blend
            rgba_low = self.dir_layers[level_int.item()](x)
            rgba_high = self.dir_layers[level_int.item() + 1](x)
            
            # Linear interpolation between scales
            rgba = rgba_low * (1 - level_frac) + rgba_high * level_frac
        else:
            # If a specific discrete level is requested
            level_idx = int(level) if isinstance(level, (int, float)) else level.item()
            level_idx = min(level_idx, self.num_scales - 1)
            rgba = self.dir_layers[level_idx](x)
            
        # Separate RGB and alpha
        rgb = rgba[..., :3]
        alpha = rgba[..., 3:4]
        
        return rgb, alpha

    def render_envmap(self, pos, t, device, level=None):
        """
        Render an environment map at a specific position, time, and scale level
        """
        dtype = self.initial_layer.weight.dtype
        cartesian = self.cartesian.to(dtype=dtype, device=device)
        
        # If level not specified, determine appropriate level based on position
        if level is None:
            # Compute distance from scene center (assuming center is at origin)
            dist = torch.norm(pos)
            
            # Map distance to level (closer = finer detail = higher level)
            # This is a simple mapping, you may want to adjust based on your scene
            # We're inverting the level so that 0 is the coarsest and num_scales-1 is the finest
            level_float = max(0, min(self.num_scales - 1, 
                                    self.num_scales - 1 - torch.log2(dist + 1).item()))
        else:
            level_float = level
            
        # Get RGB and alpha
        rgb, alpha = self.forward(pos.unsqueeze(0), cartesian.unsqueeze(0), t, level_float, device)
        
        # Shape the output as an environment map
        envmap = rgb.unsqueeze(0).permute(0, 3, 1, 2)
        alpha_map = alpha.unsqueeze(0).permute(0, 3, 1, 2)
        
        return envmap, alpha_map

    def to_image(self, x, y, z, t, device, depth_ratio, level=None):
        """
        Convert coordinates to an environment map image
        """
        # Normalize spatial coordinates
        x = 2 * (x - 55) / (512 - 2 * 55) - 1
        y = 2 * (y - 55) / (512 - 2 * 55) - 1
        z = 2 * (math.log((z)) / math.log((512 / depth_ratio)+1)) - 1
        
        # Normalize temporal coordinate
        t_normalized = 2 * (t - 1) / (30 - 1) - 1
        
        pos = torch.tensor([x, y, z], device=device, dtype=torch.float32)
        t_tensor = torch.tensor(t_normalized, device=device, dtype=torch.float32)
        
        # Render the environment map
        envmap, alpha_map = self.render_envmap(pos, t_tensor, device, level)
        
        return envmap, alpha_map
    
    def render_multiscale_envmap(self, pos, t, device):
        """
        Render an environment map using multiple scales with alpha compositing
        """
        dtype = self.initial_layer.weight.dtype
        cartesian = self.cartesian.to(dtype=dtype, device=device)
        t_tensor = torch.tensor(t, device=device, dtype=torch.float32)
        pos_tensor = pos.to(device=device, dtype=torch.float32)
        
        # Initialize accumulated RGB and transmittance
        accum_rgb = torch.zeros((1, 3, self.height, self.width), device=device)
        accum_trans = torch.ones((1, 1, self.height, self.width), device=device)
        
        # Render from coarsest to finest (back to front)
        for s in range(self.num_scales):
            # Get RGB and alpha for this scale
            rgb, alpha = self.forward(pos_tensor.unsqueeze(0), 
                                     cartesian.unsqueeze(0), 
                                     t_tensor, 
                                     float(s), 
                                     device)
            
            # Shape for compositing
            rgb = rgb.unsqueeze(0).permute(0, 3, 1, 2)  # [1, 3, H, W]
            alpha = alpha.unsqueeze(0).permute(0, 3, 1, 2)  # [1, 1, H, W]
            
            # Only consider regions relevant to this scale
            scale_weight = self.compute_scale_weight(pos_tensor, s)
            alpha = alpha * scale_weight
            
            # Composite
            accum_rgb = accum_rgb + accum_trans * alpha * rgb
            accum_trans = accum_trans * (1 - alpha)
        
        # Final environment map with background
        envmap = accum_rgb + accum_trans * torch.ones_like(accum_rgb) * 0.1  # Dark background
        
        return envmap
    
    def compute_scale_weight(self, pos, scale_idx):
        """
        Compute weight factor for a position at a specific scale level
        """
        # This is a simplified example - you should adapt this to your scene structure
        # For example, you might want to consider:
        # - Distance from scene center
        # - Distance from nearest surface
        # - Scene-specific regions of interest
        
        device = pos.device
        pos_norm = torch.norm(pos)
        
        # Each scale covers a specific range, with transitions between adjacent scales
        scale_range = self.scale_ranges[scale_idx]
        
        # Simple distance-based weighting
        if scale_idx < self.num_scales - 1:
            next_scale = self.scale_ranges[scale_idx + 1]
            # Transition region between this scale and the next
            if pos_norm < scale_range and pos_norm >= next_scale:
                # Smooth transition in the overlap region
                t = (pos_norm - next_scale) / (scale_range - next_scale)
                return torch.tensor([[[[t]]]], device=device)
            elif pos_norm >= scale_range:
                return torch.tensor([[[[1.0]]]], device=device)
            else:
                return torch.tensor([[[[0.0]]]], device=device)
        else:
            # Finest scale
            if pos_norm < scale_range:
                return torch.tensor([[[[1.0]]]], device=device)
            else:
                return torch.tensor([[[[0.0]]]], device=device)
    
    def save_exr(self, envmap_tensor, file_path):
        """
        Save environment map as EXR file
        """
        envmap_np = envmap_tensor.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)  
        imageio.imwrite(file_path, envmap_np, format='EXR')
        print(f"Saved environment map as {file_path}")

class MultiscaleLightField3DMLP(nn.Module):
    def __init__(self, hidden_dim, n_layers, num_scales=3, 
                 posenc_deg=2, direnc_deg=2, temporal_enc_deg=2, levelenc_deg=1, use_identity=False):
        super().__init__()

        # Encoders for different dimensions
        self.pos_encoder = SinusoidalEncoder(3, 0, posenc_deg, use_identity=True)
        self.dir_encoder = SinusoidalEncoder(3, 0, direnc_deg, use_identity=True)
        self.temporal_encoder = SinusoidalEncoder(1, 0, temporal_enc_deg, use_identity=True)
        self.level_encoder = SinusoidalEncoder(1, 0, levelenc_deg, use_identity=True)
        
        self.num_scales = num_scales

        # Calculate input dimensions
        pos_dim = self.pos_encoder.latent_dim
        dir_dim = self.dir_encoder.latent_dim
        temporal_dim = self.temporal_encoder.latent_dim
        level_dim = self.level_encoder.latent_dim
        
        in_dim = pos_dim + dir_dim + temporal_dim + level_dim

        # Network architecture - NeRF-like structure
        num_layers = n_layers + 1
        middle_layer_index = num_layers // 2

        self.initial_layer = nn.Linear(in_dim, hidden_dim)
        self.initial_activation = nn.LeakyReLU()

        self.pre_skip_layers = nn.ModuleList()
        for _ in range(middle_layer_index - 1):
            self.pre_skip_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.pre_skip_layers.append(nn.LeakyReLU())

        self.skip_layer = nn.Linear(hidden_dim + in_dim, hidden_dim)
        self.skip_activation = nn.LeakyReLU()

        self.post_skip_layers = nn.ModuleList()
        for _ in range(num_layers - middle_layer_index - 1):
            self.post_skip_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.post_skip_layers.append(nn.LeakyReLU())

        # Alpha (density) output
        self.alpha_layer = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
        # RGB output - one for each scale for better multi-scale representation
        self.rgb_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim // 2, 3),
                nn.Sigmoid()  # RGB values in [0,1]
            ) for _ in range(num_scales)
        ])

    def forward(self, pos, dirs, t, level):
        """
        Forward pass through the network - purely functional, no rendering logic
        
        Args:
            pos: [B, 3] position coordinates
            dirs: [B, 3] direction vectors
            t: [B] time values
            level: [B] or float, scale level (0 to num_scales-1)
            
        Returns:
            rgb: [B, 3] RGB color values
            alpha: [B, 1] Alpha/density values
        """
        # Handle batch dimension consistently
        if len(pos.shape) == 1:
            pos = pos.unsqueeze(0)
        if len(dirs.shape) == 1:
            dirs = dirs.unsqueeze(0)
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=pos.device)
        elif len(t.shape) == 0:
            t = t.unsqueeze(0)
            
        # Encode all inputs
        pos_enc = self.pos_encoder(pos)
        dirs_enc = self.dir_encoder(dirs)
        t_enc = self.temporal_encoder(t.unsqueeze(-1))
        
        # Handle level input
        if isinstance(level, (float, int)):
            level = torch.tensor([level], device=pos.device, dtype=torch.float32)
        elif not torch.is_tensor(level):
            level = torch.tensor(level, device=pos.device, dtype=torch.float32)
        if len(level.shape) == 0:
            level = level.unsqueeze(0)
            
        level_enc = self.level_encoder(level.unsqueeze(-1))
        
        # Concatenate all encoded inputs
        inputs = torch.cat([pos_enc, dirs_enc, t_enc, level_enc], dim=-1)

        # Forward pass through shared backbone
        x = self.initial_layer(inputs)
        x = self.initial_activation(x)

        original_encoded_input = inputs

        for layer in self.pre_skip_layers:
            x = layer(x)

        x = torch.cat([x, original_encoded_input], dim=-1)
        x = self.skip_layer(x)
        x = self.skip_activation(x)

        for layer in self.post_skip_layers:
            x = layer(x)
        
        # Get alpha value from shared features
        alpha = self.alpha_layer(x)
        
        # Get RGB value with scale-dependent processing
        # Convert level to float and clamp to valid range
        level_val = level.float()
        level_val = torch.clamp(level_val, 0, self.num_scales - 1)
        
        # Get integer and fractional parts for interpolation
        level_int = torch.floor(level_val).long()
        level_frac = level_val - level_int
        
        # Handle the boundary case
        level_int = torch.clamp(level_int, 0, self.num_scales - 2)
        
        # Get RGB from adjacent scales and blend
        rgb_low = self.rgb_layers[level_int[0]](x)
        rgb_high = self.rgb_layers[level_int[0] + 1](x)
        
        # Linear interpolation between scales
        rgb = rgb_low * (1 - level_frac) + rgb_high * level_frac
        
        return rgb, alpha

    def query_points(self, positions, directions, times, levels):
        """
        Batch query multiple points - helper method for external renderers
        
        Args:
            positions: [N, 3] tensor of 3D positions
            directions: [N, 3] tensor of view directions
            times: [N] tensor of time values
            levels: [N] tensor of scale levels
            
        Returns:
            rgbs: [N, 3] tensor of RGB values
            alphas: [N, 1] tensor of alpha values
        """
        # Process in batches to avoid memory issues
        batch_size = 8192  # Adjust based on GPU memory
        rgbs = []
        alphas = []
        
        for i in range(0, positions.shape[0], batch_size):
            end_idx = min(i + batch_size, positions.shape[0])
            batch_positions = positions[i:end_idx]
            batch_directions = directions[i:end_idx]
            batch_times = times[i:end_idx]
            batch_levels = levels[i:end_idx]
            
            batch_rgb, batch_alpha = self.forward(
                batch_positions, 
                batch_directions,
                batch_times, 
                batch_levels
            )
            
            rgbs.append(batch_rgb)
            alphas.append(batch_alpha)
        
        return torch.cat(rgbs, dim=0), torch.cat(alphas, dim=0)
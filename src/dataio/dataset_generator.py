# src/dataio/dataset_generator.py

import numpy as np
import torch

def heat_1d_analytical_solution(x, t, alpha=0.01):
    """
    1次元熱方程式の解析解（例: 初期条件u(x,0)=sin(pi*x), 境界条件u(0,t)=u(1,t)=0）

    u(x,t) = exp(-pi^2 * alpha * t) * sin(pi * x)

    Parameters:
    - x: np.array or torch.Tensor, 空間座標
    - t: np.array or torch.Tensor, 時間座標
    - alpha: float, 熱拡散率

    Returns:
    - u: 同じshapeの数値解
    """
    pi = np.pi
    return np.exp(-pi**2 * alpha * t) * np.sin(pi * x)


def generate_heat_1d_dataset(n_samples=1000, alpha=0.01, x_range=(0,1), t_range=(0,1), device='cpu'):
    """
    解析解に基づく1次元熱方程式の合成データセット生成

    Returns:
    - dict with keys 'x', 't', 'u' as torch.Tensor on device
    """

    x = np.random.uniform(x_range[0], x_range[1], (n_samples,1))
    t = np.random.uniform(t_range[0], t_range[1], (n_samples,1))
    u = heat_1d_analytical_solution(x, t, alpha)

    x = torch.tensor(x, dtype=torch.float32, device=device)
    t = torch.tensor(t, dtype=torch.float32, device=device)
    u = torch.tensor(u, dtype=torch.float32, device=device)

    return {'x': x, 't': t, 'u': u}

def generate_cavity_dataset(n_interior=1000, n_boundary=200, device="cpu"):
    """
    2Dキャビティフローのデータセットを生成
    出力:
        dict { 'x':..., 'y':..., 'u':..., 'v':..., 'p':..., 'mask_boundary':..., 'mask_interior':... }
    """
    # 内部点
    xi = np.random.rand(n_interior, 1)
    yi = np.random.rand(n_interior, 1)

    # 境界点（上壁 u=1, v=0 / 他の壁 u=0, v=0）
    xb = np.linspace(0, 1, int(np.sqrt(n_boundary)))
    yb = np.linspace(0, 1, int(np.sqrt(n_boundary)))
    XB, YB = np.meshgrid(xb, yb)
    boundary_coords = []
    for i in range(len(xb)):
        boundary_coords.append([xb[i], 0.0])
        boundary_coords.append([xb[i], 1.0])
        boundary_coords.append([0.0, yb[i]])
        boundary_coords.append([1.0, yb[i]])
    boundary_coords = np.unique(np.array(boundary_coords), axis=0)

    # torch変換
    x_all = torch.tensor(np.vstack([xi, boundary_coords[:,0:1]]), dtype=torch.float32, device=device)
    y_all = torch.tensor(np.vstack([yi, boundary_coords[:,1:2]]), dtype=torch.float32, device=device)

    # 初期値は未知（内部点は教師なし）
    u_all = torch.zeros_like(x_all)
    v_all = torch.zeros_like(x_all)
    p_all = torch.zeros_like(x_all)

    mask_boundary = torch.zeros(x_all.shape[0], dtype=torch.bool, device=device)
    mask_boundary[-len(boundary_coords):] = True
    mask_interior = ~mask_boundary

    return {
        "x": x_all,
        "y": y_all,
        "u": u_all,
        "v": v_all,
        "p": p_all,
        "mask_boundary": mask_boundary,
        "mask_interior": mask_interior
    }


def generate_cavity_unsteady_dataset(n_interior=2000, n_boundary=400, n_initial=400,
                                     x_range=(0.0,1.0), y_range=(0.0,1.0), t_range=(0.0,1.0),
                                     device='cpu'):
    """
    サンプリング:
      - interior points (x,y,t) for PDE residual
      - boundary points (x,y,t) for BC (including moving lid at y=1)
      - initial points (x,y,t=0) for IC (u=v=0)
    Returns dict with tensors on device.
    """
    # interior points random in space-time
    xi = np.random.uniform(x_range[0], x_range[1], (n_interior, 1))
    yi = np.random.uniform(y_range[0], y_range[1], (n_interior, 1))
    ti = np.random.uniform(t_range[0], t_range[1], (n_interior, 1))

    # boundary point generations (uniform on boundaries with random time)
    nb_side = int(math.sqrt(n_boundary))
    xb = np.linspace(x_range[0], x_range[1], nb_side)
    yb = np.linspace(y_range[0], y_range[1], nb_side)
    # collect four walls
    bcoords = []
    for x_ in xb:
        bcoords.append([x_, y_range[0]])
        bcoords.append([x_, y_range[1]])
    for y_ in yb:
        bcoords.append([x_range[0], y_])
        bcoords.append([x_range[1], y_])
    bcoords = np.unique(np.array(bcoords), axis=0)
    nb_all = bcoords.shape[0]
    tb = np.random.uniform(t_range[0], t_range[1], (nb_all, 1))

    # initial points at t=0 (uniform space)
    ni_side = int(math.sqrt(n_initial))
    xi0 = np.linspace(x_range[0], x_range[1], ni_side)
    yi0 = np.linspace(y_range[0], y_range[1], ni_side)
    X0, Y0 = np.meshgrid(xi0, yi0)
    X0 = X0.reshape(-1,1)
    Y0 = Y0.reshape(-1,1)
    t0 = np.zeros_like(X0)

    # convert to torch
    interior = {
        'x': torch.tensor(xi, dtype=torch.float32, device=device),
        'y': torch.tensor(yi, dtype=torch.float32, device=device),
        't': torch.tensor(ti, dtype=torch.float32, device=device)
    }
    boundary = {
        'x': torch.tensor(bcoords[:,0:1], dtype=torch.float32, device=device),
        'y': torch.tensor(bcoords[:,1:2], dtype=torch.float32, device=device),
        't': torch.tensor(tb, dtype=torch.float32, device=device)
    }
    initial = {
        'x': torch.tensor(X0, dtype=torch.float32, device=device),
        'y': torch.tensor(Y0, dtype=torch.float32, device=device),
        't': torch.tensor(t0, dtype=torch.float32, device=device)
    }

    # BC target values (u,v) on boundary: top lid (y=1) -> u=1, v=0; others -> u=v=0
    # We'll compute BC target when forming loss to allow time-dependent lids (here constant).
    return {'interior': interior, 'boundary': boundary, 'initial': initial}
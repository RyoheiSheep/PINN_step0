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

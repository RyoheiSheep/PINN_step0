# src/physics/heat_equation.py

import torch

def heat_1d_pde_residual(model, x, t, thermal_diffusivity):
    """
    1次元熱方程式のPDE残差を計算する関数
    ∂u/∂t = α ∂²u/∂x²

    Parameters
    ----------
    model : torch.nn.Module
        PINNモデル。入力(x,t)を受け取り温度uを出力する。
    x : torch.Tensor
        空間座標 (N,1)
    t : torch.Tensor
        時間座標 (N,1)
    thermal_diffusivity : float
        熱拡散係数 α

    Returns
    -------
    residual : torch.Tensor
        PDE残差 (N,1)
    """

    x.requires_grad_(True)
    t.requires_grad_(True)

    u = model(torch.cat([x, t], dim=1))

    # ∂u/∂t
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    # ∂u/∂x
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    # ∂²u/∂x²
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    residual = u_t - thermal_diffusivity * u_xx

    return residual

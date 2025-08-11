import torch

def navier_stokes_2d_pde_residual(model, x, y, Re):
    """
    2次元定常非圧縮性Navier-Stokes方程式のPDE残差を計算
    入力:
        model: PINNモデル、入力 (x,y) -> 出力 (u,v,p)
        x, y: torch.Tensor shape (N,1)
        Re: レイノルズ数
    出力:
        residual_u, residual_v, residual_continuity
    """

    x.requires_grad_(True)
    y.requires_grad_(True)

    uvp = model(torch.cat([x, y], dim=1))
    u = uvp[:, 0:1]
    v = uvp[:, 1:2]
    p = uvp[:, 2:3]

    # 一次導関数
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    # 二次導関数
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

    # 非線形項 u·∇u, u·∇v
    u_adv = u * u_x + v * u_y
    v_adv = u * v_x + v * v_y

    # PDE残差
    res_u = u_adv + p_x - (1.0 / Re) * (u_xx + u_yy)
    res_v = v_adv + p_y - (1.0 / Re) * (v_xx + v_yy)
    res_c = u_x + v_y  # 連続の式

    return res_u, res_v, res_c

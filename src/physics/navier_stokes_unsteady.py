# src/physics/navier_stokes_unsteady.py
import torch

def navier_stokes_2d_unsteady_residual(model, x, y, t, Re):
    """
    非定常2D非圧縮Navier-Stokes残差
    Inputs:
      model: (x,y,t) -> (u,v,p)
      x,y,t: torch.Tensor shape (N,1)
      Re: Reynolds number (scalar)
    Returns:
      res_u, res_v, res_cont  (each (N,1))
    """
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    uvp = model(torch.cat([x, y, t], dim=1))
    u = uvp[:, 0:1]
    v = uvp[:, 1:2]
    p = uvp[:, 2:3]

    # time derivatives
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    # spatial first derivatives
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

    # spatial second derivatives
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

    # nonlinear advective terms
    u_adv = u * u_x + v * u_y
    v_adv = u * v_x + v * v_y

    res_u = u_t + u_adv + p_x - (1.0 / Re) * (u_xx + u_yy)
    res_v = v_t + v_adv + p_y - (1.0 / Re) * (v_xx + v_yy)
    res_c = u_x + v_y

    return res_u, res_v, res_c

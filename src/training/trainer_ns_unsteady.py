# src/training/trainer_ns_unsteady.py
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
from ..physics.navier_stokes_unsteady import navier_stokes_2d_unsteady_residual

class TrainerNavierUnsteady:
    def __init__(self, model, dataset, physics_params, device="cpu", lr=1e-3):
        self.device = device
        self.model = model.to(device)
        self.dataset = dataset
        self.physics_params = physics_params
        self.lr = lr
        self.mse = torch.nn.MSELoss()

    def train(self, adam_epochs=2000, save_path=None, log_interval=100):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        pbar = tqdm(range(adam_epochs), desc="Training", ncols=100)

        interior = self.dataset['interior']
        boundary = self.dataset['boundary']
        initial = self.dataset['initial']

        x_i = interior['x'].to(self.device)
        y_i = interior['y'].to(self.device)
        t_i = interior['t'].to(self.device)

        x_b = boundary['x'].to(self.device)
        y_b = boundary['y'].to(self.device)
        t_b = boundary['t'].to(self.device)

        x0 = initial['x'].to(self.device)
        y0 = initial['y'].to(self.device)
        t0 = initial['t'].to(self.device)

        Re = self.physics_params.get("Re", 100.0)

        for epoch in pbar:
            optimizer.zero_grad()
            # PDE residual loss (interior points)
            res_u, res_v, res_c = navier_stokes_2d_unsteady_residual(self.model, x_i, y_i, t_i, Re)
            loss_pde = self.mse(res_u, torch.zeros_like(res_u)) + \
                       self.mse(res_v, torch.zeros_like(res_v)) + \
                       self.mse(res_c, torch.zeros_like(res_c))

            # boundary condition loss
            xybt = torch.cat([x_b, y_b, t_b], dim=1)
            uvp_b = self.model(torch.cat([x_b, y_b, t_b], dim=1))
            u_b = uvp_b[:,0:1]; v_b = uvp_b[:,1:2]

            # determine which boundary points are top lid (y==1.0) using tolerance
            y_b_np = y_b.cpu().numpy().flatten()
            top_mask = (np.isclose(y_b_np, 1.0))
            top_mask_t = torch.tensor(top_mask, dtype=torch.bool, device=self.device)
            # target for top: u=1, v=0
            if top_mask_t.any():
                loss_top = self.mse(u_b[top_mask_t], torch.ones_like(u_b[top_mask_t])) + \
                           self.mse(v_b[top_mask_t], torch.zeros_like(v_b[top_mask_t]))
            else:
                loss_top = torch.tensor(0.0, device=self.device)

            other_mask_t = ~top_mask_t
            if other_mask_t.any():
                loss_wall = self.mse(u_b[other_mask_t], torch.zeros_like(u_b[other_mask_t])) + \
                            self.mse(v_b[other_mask_t], torch.zeros_like(v_b[other_mask_t]))
            else:
                loss_wall = torch.tensor(0.0, device=self.device)

            loss_bc = loss_top + loss_wall

            # initial condition loss at t=0 (u=v=0)
            uvp0 = self.model(torch.cat([x0, y0, t0], dim=1))
            u0 = uvp0[:,0:1]; v0 = uvp0[:,1:2]
            loss_ic = self.mse(u0, torch.zeros_like(u0)) + self.mse(v0, torch.zeros_like(v0))

            # total loss (weights can be tuned)
            loss = loss_pde + loss_bc + loss_ic
            loss.backward()
            optimizer.step()

            if (epoch+1) % log_interval == 0 or epoch==0:
                pbar.set_postfix({'loss':f"{loss.item():.6e}",
                                  'pde':f"{loss_pde.item():.6e}",
                                  'bc':f"{loss_bc.item():.6e}",
                                  'ic':f"{loss_ic.item():.6e}"})

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            print(f"Saved model to {save_path}")

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.to(self.device)
        print(f"Loaded model from {filepath}")

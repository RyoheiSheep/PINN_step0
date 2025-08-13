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

    def _compute_losses(self, x_i, y_i, t_i, x_b, y_b, t_b, x0, y0, t0, Re):
        # PDE residual loss
        res_u, res_v, res_c = navier_stokes_2d_unsteady_residual(self.model, x_i, y_i, t_i, Re)
        loss_pde = self.mse(res_u, torch.zeros_like(res_u)) + \
                   self.mse(res_v, torch.zeros_like(res_v)) + \
                   self.mse(res_c, torch.zeros_like(res_c))

        # Boundary condition loss
        uvp_b = self.model(torch.cat([x_b, y_b, t_b], dim=1))
        u_b = uvp_b[:, 0:1]
        v_b = uvp_b[:, 1:2]

        y_b_np = y_b.detach().cpu().numpy().flatten()
        top_mask = np.isclose(y_b_np, 1.0)
        top_mask_t = torch.tensor(top_mask, dtype=torch.bool, device=self.device)
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

        # Initial condition loss
        uvp0 = self.model(torch.cat([x0, y0, t0], dim=1))
        u0 = uvp0[:, 0:1]
        v0 = uvp0[:, 1:2]
        loss_ic = self.mse(u0, torch.zeros_like(u0)) + self.mse(v0, torch.zeros_like(v0))

        total_loss = loss_pde + loss_bc + loss_ic
        return total_loss, loss_pde, loss_bc, loss_ic

    def train(self, adam_epochs=2000, lbfgs_epochs=500, save_path=None, log_interval=100):
        interior = self.dataset['interior']
        boundary = self.dataset['boundary']
        initial = self.dataset['initial']

        x_i, y_i, t_i = interior['x'].to(self.device), interior['y'].to(self.device), interior['t'].to(self.device)
        x_b, y_b, t_b = boundary['x'].to(self.device), boundary['y'].to(self.device), boundary['t'].to(self.device)
        x0, y0, t0 = initial['x'].to(self.device), initial['y'].to(self.device), initial['t'].to(self.device)

        Re = self.physics_params.get("Re", 100.0)

        # --- Adam phase ---
        optimizer_adam = optim.Adam(self.model.parameters(), lr=self.lr)
        pbar = tqdm(range(adam_epochs), desc="Adam Training", ncols=100)
        for epoch in pbar:
            optimizer_adam.zero_grad()
            loss, loss_pde, loss_bc, loss_ic = self._compute_losses(x_i, y_i, t_i, x_b, y_b, t_b, x0, y0, t0, Re)
            loss.backward()
            optimizer_adam.step()

            if (epoch + 1) % log_interval == 0 or epoch == 0:
                pbar.set_postfix({
                    'loss': f"{loss.item():.6e}",
                    'pde': f"{loss_pde.item():.6e}",
                    'bc': f"{loss_bc.item():.6e}",
                    'ic': f"{loss_ic.item():.6e}"
                })

        # --- L-BFGS phase ---
        optimizer_lbfgs = optim.LBFGS(self.model.parameters(),
                                      max_iter=lbfgs_epochs,
                                      tolerance_grad=1e-8,
                                      tolerance_change=1e-9,
                                      history_size=50)

        iteration = 0
        def closure():
            nonlocal iteration
            optimizer_lbfgs.zero_grad()
            loss, loss_pde, loss_bc, loss_ic = self._compute_losses(x_i, y_i, t_i, x_b, y_b, t_b, x0, y0, t0, Re)
            loss.backward()
            iteration += 1
            if iteration % 10 == 0:
                print(f"[L-BFGS {iteration}] loss={loss.item():.6e}, pde={loss_pde.item():.6e}, bc={loss_bc.item():.6e}, ic={loss_ic.item():.6e}")
            return loss

        print(f"Starting L-BFGS optimization for up to {lbfgs_epochs} iterations...")
        optimizer_lbfgs.step(closure)
        print("L-BFGS optimization finished.")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            print(f"Saved model to {save_path}")

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.to(self.device)
        print(f"Loaded model from {filepath}")

# src/training/trainer.py

import torch
import torch.optim as optim
import os
from tqdm import tqdm

class Trainer:
    def __init__(self, model, physics_residual_func, dataset, physics_params, device='cpu'):
        """
        Parameters:
        - model: PINNモデル
        - physics_residual_func: PDE残差計算関数
        - dataset: dict {'x': tensor, 't': tensor, 'u': tensor}
        - physics_params: dict, 物理パラメータ
        - device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model = model.to(device)
        self.physics_residual_func = physics_residual_func
        self.dataset = dataset
        self.physics_params = physics_params

        self.x = dataset['x'].to(device)
        self.t = dataset['t'].to(device)
        self.u = dataset['u'].to(device)

        # 損失関数はMSE
        self.mse_loss = torch.nn.MSELoss()

    def train(self, adam_epochs=1000, lbfgs_epochs=500, save_path=None):
        # Adamオプティマイザ
        optimizer_adam = optim.Adam(self.model.parameters(), lr=1e-3)

        print(f"Starting training with Adam optimizer for {adam_epochs} epochs.")
        pbar = tqdm(range(adam_epochs), desc="Adam Training", ncols=80)
        for epoch in pbar:
            optimizer_adam.zero_grad()

            u_pred = self.model(torch.cat([self.x, self.t], dim=1))
            mse_data = self.mse_loss(u_pred, self.u)

            residual = self.physics_residual_func(self.model, self.x, self.t, self.physics_params)
            mse_pde = self.mse_loss(residual, torch.zeros_like(residual))

            loss = mse_data + mse_pde
            loss.backward()
            optimizer_adam.step()

            pbar.set_postfix(loss=f"{loss.item():.6f}", data_loss=f"{mse_data.item():.6f}", pde_loss=f"{mse_pde.item():.6f}")

        # L-BFGSオプティマイザ
        optimizer_lbfgs = optim.LBFGS(self.model.parameters(), max_iter=lbfgs_epochs, tolerance_grad=1e-8, tolerance_change=1e-9)

        print(f"Starting training with L-BFGS optimizer for up to {lbfgs_epochs} iterations.")

        iteration = 0
        def closure():
            nonlocal iteration
            optimizer_lbfgs.zero_grad()
            u_pred = self.model(torch.cat([self.x, self.t], dim=1))
            mse_data = self.mse_loss(u_pred, self.u)
            residual = self.physics_residual_func(self.model, self.x, self.t, **self.physics_params)
            mse_pde = self.mse_loss(residual, torch.zeros_like(residual))
            loss = mse_data + mse_pde
            loss.backward()

            iteration += 1
            print(f"[L-BFGS Iter {iteration}] Loss: {loss.item():.6f}, Data Loss: {mse_data.item():.6f}, PDE Loss: {mse_pde.item():.6f}")

            return loss

        optimizer_lbfgs.step(closure)

        print("L-BFGS optimization finished.")

        if save_path:
            self.save_model(save_path)

    def save_model(self, filepath):
        """モデルのパラメータを保存"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        print(f"モデルを保存しました: {filepath}")

    def load_model(self, filepath):
        """モデルのパラメータを読み込み"""
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.to(self.device)
        print(f"モデルを読み込みました: {filepath}")

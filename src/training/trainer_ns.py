import torch
import torch.optim as optim
from tqdm import tqdm

class TrainerNavier:
    def __init__(self, model, physics_residual_func, dataset, physics_params, device="cpu"):
        self.device = device
        self.model = model.to(device)
        self.physics_residual_func = physics_residual_func
        self.dataset = dataset
        self.physics_params = physics_params
        self.mse_loss = torch.nn.MSELoss()

    def train(self, adam_epochs=1000, save_path=None):
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        pbar = tqdm(range(adam_epochs), desc="Training", ncols=80)

        x = self.dataset["x"].to(self.device)
        y = self.dataset["y"].to(self.device)
        mask_b = self.dataset["mask_boundary"].to(self.device)
        mask_i = self.dataset["mask_interior"].to(self.device)

        for epoch in pbar:
            optimizer.zero_grad()
            uvp = self.model(torch.cat([x, y], dim=1))
            u_pred, v_pred, p_pred = uvp[:,0:1], uvp[:,1:2], uvp[:,2:3]

            # PDE残差（内部点のみ）
            res_u, res_v, res_c = self.physics_residual_func(
                self.model, x[mask_i], y[mask_i], self.physics_params["Re"]
            )
            loss_pde = self.mse_loss(res_u, torch.zeros_like(res_u)) \
                     + self.mse_loss(res_v, torch.zeros_like(res_v)) \
                     + self.mse_loss(res_c, torch.zeros_like(res_c))

            # 境界条件損失
            u_b = u_pred[mask_b]
            v_b = v_pred[mask_b]
            x_b = x[mask_b]
            y_b = y[mask_b]

            # 上壁 (y=1): u=1, v=0
            top_mask = (y_b == 1.0)
            loss_top = self.mse_loss(u_b[top_mask], torch.ones_like(u_b[top_mask])) \
                     + self.mse_loss(v_b[top_mask], torch.zeros_like(v_b[top_mask]))

            # 他の壁: u=0, v=0
            other_mask = ~top_mask
            loss_wall = self.mse_loss(u_b[other_mask], torch.zeros_like(u_b[other_mask])) \
                      + self.mse_loss(v_b[other_mask], torch.zeros_like(v_b[other_mask]))

            loss_bc = loss_top + loss_wall

            loss = loss_pde + loss_bc
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.6f}", pde=f"{loss_pde.item():.6f}", bc=f"{loss_bc.item():.6f}")

        if save_path:
            torch.save(self.model.state_dict(), save_path)
            print(f"モデル保存: {save_path}")

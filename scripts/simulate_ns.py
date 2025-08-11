import torch
import numpy as np
import matplotlib.pyplot as plt
from models.pinn_navier2d import PINN_Navier2D
from configs import physics_params_navier2d

# --- 設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./models_ckpt/navier2d_trained.pth"  # 学習済みモデル
output_dir = "./outputs/navier2d_inference"
t_infer = 1.0   # 推論時刻

# グリッド解像度
nx, ny = 64, 64

# --- モデル読み込み ---
model = PINN_Navier2D(physics_params_navier2d.params).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# --- 座標グリッド作成 ---
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)
T = np.full_like(X, t_infer)

coords = np.stack([X.flatten(), Y.flatten(), T.flatten()], axis=-1)
coords_tensor = torch.tensor(coords, dtype=torch.float32).to(device)

# --- 推論 ---
with torch.no_grad():
    pred = model(coords_tensor)
    u_pred = pred[:, 0].cpu().numpy().reshape(ny, nx)
    v_pred = pred[:, 1].cpu().numpy().reshape(ny, nx)
    p_pred = pred[:, 2].cpu().numpy().reshape(ny, nx)

# --- 外部データから解析解を読み込む例（今はコメントアウト） ---
# true_data = np.load("./data/navier2d_true_t1.0.npy", allow_pickle=True).item()
# u_true = true_data["u"]
# v_true = true_data["v"]
# p_true = true_data["p"]

# --- 保存 ---
import os
os.makedirs(output_dir, exist_ok=True)
np.save(f"{output_dir}/u_pred_t{t_infer}.npy", u_pred)
np.save(f"{output_dir}/v_pred_t{t_infer}.npy", v_pred)
np.save(f"{output_dir}/p_pred_t{t_infer}.npy", p_pred)

# --- 可視化 ---
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
im0 = axs[0].imshow(u_pred, origin="lower", extent=[0, 1, 0, 1])
axs[0].set_title("u velocity")
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(v_pred, origin="lower", extent=[0, 1, 0, 1])
axs[1].set_title("v velocity")
plt.colorbar(im1, ax=axs[1])

im2 = axs[2].imshow(p_pred, origin="lower", extent=[0, 1, 0, 1])
axs[2].set_title("Pressure")
plt.colorbar(im2, ax=axs[2])

plt.tight_layout()
plt.savefig(f"{output_dir}/navier2d_pred_t{t_infer}.png", dpi=200)
plt.close()

print(f"推論結果を {output_dir} に保存しました。")
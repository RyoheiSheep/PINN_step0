# scripts/simulate_ns.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt

from configs import physics_params_navier2d
from src.models.pinn_ns import PINNNavier2D

def simulate(model_path, save_dir="simulation_results_ns", device='cpu'):
    device = torch.device(device)
    os.makedirs(save_dir, exist_ok=True)

    # モデル構築
    model = PINNNavier2D(input_dim=2, output_dim=3, hidden_layers=[50, 50, 50])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # グリッド設定（2D空間）
    nx, ny = 64, 64
    x_vals = np.linspace(physics_params_navier2d.params["x_min"], physics_params_navier2d.params["x_max"], nx)
    y_vals = np.linspace(physics_params_navier2d.params["y_min"], physics_params_navier2d.params["y_max"], ny)

    X, Y = np.meshgrid(x_vals, y_vals)
    xy_flat = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1),
                           dtype=torch.float32, device=device)

    # PINN予測
    with torch.no_grad():
        pred = model(xy_flat)
    u_pred = pred[:, 0].cpu().numpy().reshape(X.shape)
    v_pred = pred[:, 1].cpu().numpy().reshape(X.shape)
    p_pred = pred[:, 2].cpu().numpy().reshape(X.shape)

    # --- 外部データから解析解を読み込む例（今はコメントアウト） ---
    # true_data = np.load("./data/navier2d_true.npy", allow_pickle=True).item()
    # u_true = true_data["u"]
    # v_true = true_data["v"]
    # p_true = true_data["p"]

    # 結果保存
    np.savez(os.path.join(save_dir, "simulation_data.npz"),
             x=X, y=Y, u_pred=u_pred, v_pred=v_pred, p_pred=p_pred)
    print(f"シミュレーション結果を保存しました: {os.path.join(save_dir, 'simulation_data.npz')}")

    # 可視化
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axs[0].imshow(u_pred, origin="lower",
                        extent=[physics_params_navier2d.params["x_min"], physics_params_navier2d.params["x_max"],
                                physics_params_navier2d.params["y_min"], physics_params_navier2d.params["y_max"]])
    axs[0].set_title("u velocity")
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(v_pred, origin="lower",
                        extent=[physics_params_navier2d.params["x_min"], physics_params_navier2d.params["x_max"],
                                physics_params_navier2d.params["y_min"], physics_params_navier2d.params["y_max"]])
    axs[1].set_title("v velocity")
    plt.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(p_pred, origin="lower",
                        extent=[physics_params_navier2d.params["x_min"], physics_params_navier2d.params["x_max"],
                                physics_params_navier2d.params["y_min"], physics_params_navier2d.params["y_max"]])
    axs[2].set_title("Pressure")
    plt.colorbar(im2, ax=axs[2])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fields_plot.png"), dpi=200)
    plt.close()
    print(f"可視化画像を保存しました: {os.path.join(save_dir, 'fields_plot.png')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="学習済みモデルのファイルパス")
    parser.add_argument("--save_dir", default="simulation_results_ns", help="結果保存ディレクトリ")
    parser.add_argument("--device", default="cpu", help="実行デバイス(cpu or cuda)")
    args = parser.parse_args()

    simulate(args.model_path, args.save_dir, args.device)

# scripts/simulate_heat.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from configs import physics_params_heat
from src.models.pinn_heat import PINNHeat1D

def simulate(model_path, save_dir="simulation_results", device='cpu'):
    device = torch.device(device)
    os.makedirs(save_dir, exist_ok=True)

    # モデル構築
    model = PINNHeat1D(input_dim=2, output_dim=1, hidden_layers=[50, 50, 50])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # シミュレーション用に格子を作成
    x_vals = np.linspace(physics_params_heat.params["x_min"], physics_params_heat.params["x_max"], 100)
    t_vals = np.linspace(physics_params_heat.params["t_min"], physics_params_heat.params["t_max"], 100)

    X, T = np.meshgrid(x_vals, t_vals)
    x_flat = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    t_flat = torch.tensor(T.flatten(), dtype=torch.float32, device=device).unsqueeze(1)

    with torch.no_grad():
        u_pred = model(torch.cat([x_flat, t_flat], dim=1))
    U = u_pred.cpu().numpy().reshape(X.shape)

    # 結果をnpzで保存
    np.savez(os.path.join(save_dir, "simulation_data.npz"), x=X, t=T, u=U)
    print(f"シミュレーション結果を保存しました: {os.path.join(save_dir, 'simulation_data.npz')}")

    # 結果の可視化（例: t固定でx方向の温度分布）
    import matplotlib.pyplot as plt
    for idx, t_val in enumerate([0, 25, 50, 75, 99]):  # 時刻を5点抜粋
        plt.plot(X[t_val, :], U[t_val, :], label=f"t={T[t_val,0]:.2f}")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.legend()
    plt.title("PINN Heat 1D Simulation Results")
    plt.savefig(os.path.join(save_dir, "simulation_plot.png"))
    plt.close()
    print(f"プロットを保存しました: {os.path.join(save_dir, 'simulation_plot.png')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="学習済みモデルのファイルパス")
    parser.add_argument("--save_dir", default="simulation_results", help="結果保存ディレクトリ")
    parser.add_argument("--device", default="cpu", help="実行デバイス(cpu or cuda)")
    args = parser.parse_args()

    simulate(args.model_path, args.save_dir, args.device)

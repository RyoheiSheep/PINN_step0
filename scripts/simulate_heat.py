# scripts/simulate_heat.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt

from configs import physics_params_heat
from src.models.pinn_heat import PINNHeat1D

def analytical_solution(X, T, alpha=physics_params_heat.params["alpha"]):
    """
    境界条件 u(0,t)=u(L,t)=0, 初期条件 u(x,0) = sin(pi x)
    L=1 と仮定
    """
    return np.exp(-(np.pi**2) * alpha * T) * np.sin(np.pi * X)

def simulate(model_path, save_dir="simulation_results", device='cpu'):
    device = torch.device(device)
    os.makedirs(save_dir, exist_ok=True)

    # モデル構築
    model = PINNHeat1D(input_dim=2, output_dim=1, hidden_layers=[50, 50, 50])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 格子生成
    x_vals = np.linspace(physics_params_heat.params["x_min"], physics_params_heat.params["x_max"], 100)
    t_vals = np.linspace(physics_params_heat.params["t_min"], physics_params_heat.params["t_max"], 100)

    X, T = np.meshgrid(x_vals, t_vals)
    x_flat = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    t_flat = torch.tensor(T.flatten(), dtype=torch.float32, device=device).unsqueeze(1)

    # PINN予測
    with torch.no_grad():
        u_pred = model(torch.cat([x_flat, t_flat], dim=1))
    U_pred = u_pred.cpu().numpy().reshape(X.shape)

    # 解析解
    U_analytical = analytical_solution(X, T)

    # 結果保存
    np.savez(os.path.join(save_dir, "simulation_data.npz"),
             x=X, t=T, u_pred=U_pred, u_analytical=U_analytical)
    print(f"シミュレーション結果を保存しました: {os.path.join(save_dir, 'simulation_data.npz')}")

    # 比較プロット（例: 5つの時刻でx方向比較）
    for idx in [0, 25, 50, 75, 99]:
        plt.plot(X[idx, :], U_pred[idx, :], 'r-', label="PINN" if idx == 0 else "")
        plt.plot(X[idx, :], U_analytical[idx, :], 'k--', label="Analytical" if idx == 0 else "")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.legend()
    plt.title("PINN vs Analytical Solution")
    plt.savefig(os.path.join(save_dir, "comparison_plot.png"))
    plt.close()
    print(f"比較プロットを保存しました: {os.path.join(save_dir, 'comparison_plot.png')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="学習済みモデルのファイルパス")
    parser.add_argument("--save_dir", default="simulation_results", help="結果保存ディレクトリ")
    parser.add_argument("--device", default="cpu", help="実行デバイス(cpu or cuda)")
    args = parser.parse_args()

    simulate(args.model_path, args.save_dir, args.device)

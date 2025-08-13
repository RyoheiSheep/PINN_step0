# scripts/simulate_ns_unsteady.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio

from src.models.pinn_ns import PINNNavier2D
from configs import physics_params_ns  # expect x_min,x_max,y_min,y_max,t_min,t_max

def simulate_and_save(model_path, save_dir="simulation_ns_unsteady", device='cpu', nx=64, ny=64, nt=50):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device)

    # load model
    model = PINNNavier2D(input_dim=3, output_dim=3, hidden_layers=[128,128,128,128])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 空間・時間のグリッド
    x_vals = np.linspace(physics_params_ns.params.get("x_min",0.0), physics_params_ns.params.get("x_max",1.0), nx)
    y_vals = np.linspace(physics_params_ns.params.get("y_min",0.0), physics_params_ns.params.get("y_max",1.0), ny)
    t_vals = np.linspace(physics_params_ns.params.get("t_min",0.0), physics_params_ns.params.get("t_max",1.0), nt)

    Xg, Yg = np.meshgrid(x_vals, y_vals)
    filenames = []

    for it, t in enumerate(t_vals):
        XYT = np.stack([Xg.flatten(), Yg.flatten(), np.full(Xg.size, t)], axis=1)
        with torch.no_grad():
            pred = model(torch.tensor(XYT, dtype=torch.float32, device=device))
        u = pred[:,0].cpu().numpy().reshape(Yg.shape)
        v = pred[:,1].cpu().numpy().reshape(Yg.shape)
        p = pred[:,2].cpu().numpy().reshape(Yg.shape)

        # 保存（npz）
        np.savez(os.path.join(save_dir, f"frame_{it:04d}.npz"), x=Xg, y=Yg, u=u, v=v, p=p, t=t)

        # 速度の大きさ
        speed = np.sqrt(u**2 + v**2)

        # 可視化（背景に速度の絶対値、上に流線図）
        fig, ax = plt.subplots(figsize=(4,4))
        im = ax.imshow(speed, origin='lower',
                       extent=[x_vals[0], x_vals[-1], y_vals[0], y_vals[-1]],
                       cmap='viridis')
        # 流線図
        ax.streamplot(x_vals, y_vals, u, v, color='white', density=1.0, linewidth=0.5)

        ax.set_title(f"t={t:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, label="|velocity|")
        plt.tight_layout()

        # PNG保存
        png_path = os.path.join(save_dir, f"frame_{it:04d}.png")
        fig.savefig(png_path, dpi=100)
        plt.close(fig)

        filenames.append(png_path)

    # GIF作成
    gif_path = os.path.join(save_dir, "flow_evolution.gif")
    with imageio.get_writer(gif_path, mode='I', duration=0.2) as writer:
        for fname in filenames:
            image = imageio.imread(fname)
            writer.append_data(image)
    print(f"Saved GIF: {gif_path}")

    # 最終フレームPNG
    final_png = filenames[-1]
    final_png_out = os.path.join(save_dir, "final_frame.png")
    os.replace(final_png, final_png_out)
    print(f"Saved final png: {final_png_out}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--save_dir", default="simulation_ns_unsteady")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nt", type=int, default=50)
    args = parser.parse_args()

    simulate_and_save(args.model_path, args.save_dir, args.device, nx=args.nx, ny=args.ny, nt=args.nt)

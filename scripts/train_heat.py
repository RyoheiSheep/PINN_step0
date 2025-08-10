# scripts/train_heat.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from configs import physics_params_heat
from src.models.pinn_heat import PINNHeat1D
from src.physics.heat_equation import heat_1d_pde_residual
from src.dataio.dataset_generator import generate_heat_1d_dataset
from src.training.trainer import Trainer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # データ生成
    dataset = generate_heat_1d_dataset(
        n_samples=2000,
        alpha=physics_params_heat.params["thermal_diffusivity"],
        x_range=(physics_params_heat.params["x_min"], physics_params_heat.params["x_max"]),
        t_range=(physics_params_heat.params["t_min"], physics_params_heat.params["t_max"]),
        device=device
    )

    # モデル構築
    model = PINNHeat1D(input_dim=2, output_dim=1, hidden_layers=[50, 50, 50])

    # トレーナーセットアップ
    trainer = Trainer(model, heat_1d_pde_residual, dataset, physics_params_heat.params["thermal_diffusivity"], device=device)

    # 学習実行
    trainer.train(adam_epochs=1000, lbfgs_epochs=500, save_path="trained_models/pinn_heat1d.pth")
    
    # train_heat.py の最後の train 呼び出しを以下のように変更例


if __name__ == "__main__":
    main()

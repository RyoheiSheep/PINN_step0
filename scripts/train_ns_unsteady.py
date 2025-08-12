# scripts/train_ns_unsteady.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.models.pinn_ns import PINNNavier2D
from src.dataio.dataset_generator import generate_cavity_unsteady_dataset
from src.training.trainer_ns_unsteady import TrainerNavierUnsteady

physics_params = {"Re": 100.0}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = generate_cavity_unsteady_dataset(n_interior=4000, n_boundary=400, n_initial=400,
                                               x_range=(0.0,1.0), y_range=(0.0,1.0), t_range=(0.0,1.0),
                                               device=device)
    model = PINNNavier2D(input_dim=3, output_dim=3, hidden_layers=[128, 128, 128])  # deeper net for complexity

    trainer = TrainerNavierUnsteady(model, dataset, physics_params, device=device, lr=1e-3)
    trainer.train(adam_epochs=5000, save_path="trained_models/pinn_navier2d_unsteady.pth", log_interval=50)

if __name__ == "__main__":
    main()

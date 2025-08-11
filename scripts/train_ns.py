import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.models.pinn_ns import PINNNavier2D
from src.physics.navier_stokes import navier_stokes_2d_pde_residual
from src.dataio.dataset_generator import generate_cavity_dataset
from src.training.trainer_ns import TrainerNavier

physics_params = {
    "Re": 100.0
}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = generate_cavity_dataset(n_interior=2000, n_boundary=400, device=device)
    model = PINNNavier2D(input_dim=2, output_dim=3, hidden_layers=[50, 50, 50])

    trainer = TrainerNavier(model, navier_stokes_2d_pde_residual, dataset, physics_params, device=device)
    trainer.train(adam_epochs=2000, save_path="trained_models/pinn_navier2d.pth")

if __name__ == "__main__":
    main()

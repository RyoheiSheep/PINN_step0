from .base_pinn import BasePINN

class PINNNavier2D(BasePINN):
    def __init__(self, **kwargs):
        super(PINNNavier2D, self).__init__(**kwargs)

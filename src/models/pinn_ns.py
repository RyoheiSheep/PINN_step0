from .base_pinn import BasePINN

class PINNHeat1D(BasePINN):
    def __init__(self, **kwargs):
        super(PINNHeat1D, self).__init__(**kwargs)

from .losses import (CharbonnierLoss, GANLoss, L1Loss, MSELoss, PerceptualLoss,
                     WeightedTVLoss, g_path_regularize, compute_gradient_penalty,
                     r1_penalty)

__all__ = [
    'L1Loss', 'MSELoss', 'CharbonnierLoss', 'WeightedTVLoss', 'PerceptualLoss',
    'GANLoss', 'compute_gradient_penalty', 'r1_penalty', 'g_path_regularize'
]

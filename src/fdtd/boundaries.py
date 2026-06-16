import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import json

EPS0 = 8.854e-12
MU0 = 4 * np.pi * 1e-7
C0 = 1 / np.sqrt(EPS0 * MU0)
ETA0 = np.sqrt(MU0 / EPS0)


@dataclass
class BoundaryCondition:
    x_min_type: str = 'mur'
    x_max_type: str = 'mur'
    y_min_type: str = 'mur'
    y_max_type: str = 'mur'
    pml_layers: int = 8

    def to_dict(self):
        return {
            'x_min_type': self.x_min_type,
            'x_max_type': self.x_max_type,
            'y_min_type': self.y_min_type,
            'y_max_type': self.y_max_type,
            'pml_layers': self.pml_layers
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            x_min_type=data.get('x_min_type', 'mur'),
            x_max_type=data.get('x_max_type', 'mur'),
            y_min_type=data.get('y_min_type', 'mur'),
            y_max_type=data.get('y_max_type', 'mur'),
            pml_layers=data.get('pml_layers', 8)
        )

    def get_pml_sizes(self):
        x_min_pml = self.pml_layers if self.x_min_type == 'pml' else 0
        x_max_pml = self.pml_layers if self.x_max_type == 'pml' else 0
        y_min_pml = self.pml_layers if self.y_min_type == 'pml' else 0
        y_max_pml = self.pml_layers if self.y_max_type == 'pml' else 0
        return x_min_pml, x_max_pml, y_min_pml, y_max_pml


class CPML:
    def __init__(self, nx: int, ny: int, dx: float, dy: float, dt: float,
                 bc: BoundaryCondition):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.bc = bc

        x_min_pml, x_max_pml, y_min_pml, y_max_pml = bc.get_pml_sizes()
        self.x_min_pml = x_min_pml
        self.x_max_pml = x_max_pml
        self.y_min_pml = y_min_pml
        self.y_max_pml = y_max_pml

        self.has_pml = (x_min_pml > 0 or x_max_pml > 0 or
                       y_min_pml > 0 or y_max_pml > 0)

    def update_psi_from_e(self, ez: np.ndarray):
        pass

    def update_psi_from_h(self, hx: np.ndarray, hy: np.ndarray):
        pass

    def apply_h(self, hx: np.ndarray, hy: np.ndarray, ez: np.ndarray):
        pass

    def apply_e(self, ez: np.ndarray, hx: np.ndarray, hy: np.ndarray):
        pass

    def apply_pec(self, ez: np.ndarray, hx: np.ndarray, hy: np.ndarray):
        if self.bc.x_min_type == 'pec':
            ez[0, :] = 0
            hx[0, :] = 0
        if self.bc.x_max_type == 'pec':
            ez[-1, :] = 0
            hx[-1, :] = 0
        if self.bc.y_min_type == 'pec':
            ez[:, 0] = 0
            hy[:, 0] = 0
        if self.bc.y_max_type == 'pec':
            ez[:, -1] = 0
            hy[:, -1] = 0

    def apply_mur(self, ez: np.ndarray, ez_prev: np.ndarray,
                  dt: float, dx: float, dy: float):
        if self.has_pml:
            return

        c = C0
        mur_coef_x = (c * dt - dx) / (c * dt + dx)
        mur_coef_y = (c * dt - dy) / (c * dt + dy)

        if self.bc.x_min_type == 'mur':
            ez[0, 1:-1] = ez_prev[1, 1:-1] + mur_coef_x * (ez[1, 1:-1] - ez_prev[0, 1:-1])
        if self.bc.x_max_type == 'mur':
            ez[-1, 1:-1] = ez_prev[-2, 1:-1] + mur_coef_x * (ez[-2, 1:-1] - ez_prev[-1, 1:-1])
        if self.bc.y_min_type == 'mur':
            ez[1:-1, 0] = ez_prev[1:-1, 1] + mur_coef_y * (ez[1:-1, 1] - ez_prev[1:-1, 0])
        if self.bc.y_max_type == 'mur':
            ez[1:-1, -1] = ez_prev[1:-1, -2] + mur_coef_y * (ez[1:-1, -2] - ez_prev[1:-1, -1])

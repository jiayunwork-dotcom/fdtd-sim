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

        self.psi_e_x = np.zeros((nx, ny))
        self.psi_e_y = np.zeros((nx, ny))
        self.psi_h_x = np.zeros((nx, ny))
        self.psi_h_y = np.zeros((nx, ny))

        self._init_coefficients()

    def _init_coefficients(self):
        m = 4
        sigma_max = 0.8 * (m + 1) / (ETA0 * self.dx)
        alpha_max = 0.0
        kappa_max = 5.0

        self.kappa_e_x = np.ones(self.nx)
        self.alpha_e_x = np.zeros(self.nx)
        self.b_e_x = np.ones(self.nx)
        self.c_e_x = np.zeros(self.nx)

        self.kappa_m_x = np.ones(self.nx)
        self.alpha_m_x = np.zeros(self.nx)
        self.b_m_x = np.ones(self.nx)
        self.c_m_x = np.zeros(self.nx)

        for i in range(self.nx):
            dist = 0
            if i < self.x_min_pml:
                dist = (self.x_min_pml - i) / max(self.x_min_pml, 1)
            elif i >= self.nx - self.x_max_pml:
                dist = (i - (self.nx - self.x_max_pml - 1)) / max(self.x_max_pml, 1)

            if dist > 0:
                sigma = sigma_max * (dist ** m)
                alpha = alpha_max * (1 - dist)
                kappa = 1 + (kappa_max - 1) * (dist ** m)

                self.kappa_e_x[i] = kappa
                self.alpha_e_x[i] = alpha
                self.b_e_x[i] = np.exp(-(sigma / kappa + alpha) * self.dt / EPS0)
                self.c_e_x[i] = sigma / (kappa * (sigma + kappa * alpha)) * (self.b_e_x[i] - 1)

                sigma_m = sigma * MU0 / EPS0
                self.kappa_m_x[i] = kappa
                self.alpha_m_x[i] = alpha
                self.b_m_x[i] = np.exp(-(sigma_m / kappa + alpha) * self.dt / MU0)
                self.c_m_x[i] = sigma_m / (kappa * (sigma_m + kappa * alpha)) * (self.b_m_x[i] - 1)

        self.kappa_e_y = np.ones(self.ny)
        self.alpha_e_y = np.zeros(self.ny)
        self.b_e_y = np.ones(self.ny)
        self.c_e_y = np.zeros(self.ny)

        self.kappa_m_y = np.ones(self.ny)
        self.alpha_m_y = np.zeros(self.ny)
        self.b_m_y = np.ones(self.ny)
        self.c_m_y = np.zeros(self.ny)

        for j in range(self.ny):
            dist = 0
            if j < self.y_min_pml:
                dist = (self.y_min_pml - j) / max(self.y_min_pml, 1)
            elif j >= self.ny - self.y_max_pml:
                dist = (j - (self.ny - self.y_max_pml - 1)) / max(self.y_max_pml, 1)

            if dist > 0:
                sigma = sigma_max * (dist ** m)
                alpha = alpha_max * (1 - dist)
                kappa = 1 + (kappa_max - 1) * (dist ** m)

                self.kappa_e_y[j] = kappa
                self.alpha_e_y[j] = alpha
                self.b_e_y[j] = np.exp(-(sigma / kappa + alpha) * self.dt / EPS0)
                self.c_e_y[j] = sigma / (kappa * (sigma + kappa * alpha)) * (self.b_e_y[j] - 1)

                sigma_m = sigma * MU0 / EPS0
                self.kappa_m_y[j] = kappa
                self.alpha_m_y[j] = alpha
                self.b_m_y[j] = np.exp(-(sigma_m / kappa + alpha) * self.dt / MU0)
                self.c_m_y[j] = sigma_m / (kappa * (sigma_m + kappa * alpha)) * (self.b_m_y[j] - 1)

    def update_psi_from_e(self, ez: np.ndarray):
        de_x = np.zeros_like(ez)
        de_y = np.zeros_like(ez)

        de_x[1:-1, :] = (ez[2:, :] - ez[:-2, :]) / (2 * self.dx)
        de_y[:, 1:-1] = (ez[:, 2:] - ez[:, :-2]) / (2 * self.dy)

        b_m_x = self.b_m_x[:, np.newaxis]
        c_m_x = self.c_m_x[:, np.newaxis]
        b_m_y = self.b_m_y[np.newaxis, :]
        c_m_y = self.c_m_y[np.newaxis, :]

        self.psi_e_x = b_m_x * self.psi_e_x + c_m_x * de_x
        self.psi_e_y = b_m_y * self.psi_e_y + c_m_y * de_y

    def update_psi_from_h(self, hx: np.ndarray, hy: np.ndarray):
        dhx_y = np.zeros_like(hx)
        dhy_x = np.zeros_like(hy)

        if hx.shape[1] > 2:
            dhx_y[:, 1:-1] = (hx[:, 2:] - hx[:, :-2]) / (2 * self.dy)
        if hy.shape[0] > 2:
            dhy_x[1:-1, :] = (hy[2:, :] - hy[:-2, :]) / (2 * self.dx)

        b_e_x = self.b_e_x[:, np.newaxis]
        c_e_x = self.c_e_x[:, np.newaxis]
        b_e_y = self.b_e_y[np.newaxis, :]
        c_e_y = self.c_e_y[np.newaxis, :]

        self.psi_h_x = b_e_x * self.psi_h_x + c_e_x * dhy_x
        self.psi_h_y = b_e_y * self.psi_h_y + c_e_y * dhx_y

    def apply_h(self, hx: np.ndarray, hy: np.ndarray, ez: np.ndarray):
        de_x = np.zeros_like(ez)
        de_y = np.zeros_like(ez)

        de_x[1:-1, :] = (ez[2:, :] - ez[:-2, :]) / (2 * self.dx)
        de_y[:, 1:-1] = (ez[:, 2:] - ez[:, :-2]) / (2 * self.dy)

        kappa_m_x = self.kappa_m_x[:, np.newaxis]
        kappa_m_y = self.kappa_m_y[np.newaxis, :]

        hx[:, :] = hx[:, :] + self.dt / MU0 * (
            de_y * (1 - 1/kappa_m_y) - self.psi_e_y
        )

        hy[:, :] = hy[:, :] - self.dt / MU0 * (
            de_x * (1 - 1/kappa_m_x) - self.psi_e_x
        )

    def apply_e(self, ez: np.ndarray, hx: np.ndarray, hy: np.ndarray):
        dhx_y = np.zeros_like(hx)
        dhy_x = np.zeros_like(hy)

        if hx.shape[1] > 2:
            dhx_y[:, 1:-1] = (hx[:, 2:] - hx[:, :-2]) / (2 * self.dy)
        if hy.shape[0] > 2:
            dhy_x[1:-1, :] = (hy[2:, :] - hy[:-2, :]) / (2 * self.dx)

        kappa_e_x = self.kappa_e_x[:, np.newaxis]
        kappa_e_y = self.kappa_e_y[np.newaxis, :]

        ez[:, :] = ez[:, :] + self.dt / EPS0 * (
            -dhy_x * (1 - 1/kappa_e_x) + self.psi_h_x
            + dhx_y * (1 - 1/kappa_e_y) - self.psi_h_y
        )

    def apply_pec(self, ez: np.ndarray, hx: np.ndarray, hy: np.ndarray):
        if self.bc.x_min_type == 'pec':
            ez[0, :] = 0
        if self.bc.x_max_type == 'pec':
            ez[-1, :] = 0
        if self.bc.y_min_type == 'pec':
            ez[:, 0] = 0
        if self.bc.y_max_type == 'pec':
            ez[:, -1] = 0

    def apply_mur(self, ez: np.ndarray, ez_prev: np.ndarray, dt: float, dx: float, dy: float):
        c = C0
        coeff_x = (c * dt - dx) / (c * dt + dx)
        coeff_y = (c * dt - dy) / (c * dt + dy)

        if self.bc.x_min_type == 'mur':
            ez[0, 1:-1] = ez_prev[1, 1:-1] + coeff_x * (ez[1, 1:-1] - ez_prev[0, 1:-1])
        if self.bc.x_max_type == 'mur':
            ez[-1, 1:-1] = ez_prev[-2, 1:-1] + coeff_x * (ez[-2, 1:-1] - ez_prev[-1, 1:-1])
        if self.bc.y_min_type == 'mur':
            ez[1:-1, 0] = ez_prev[1:-1, 1] + coeff_y * (ez[1:-1, 1] - ez_prev[1:-1, 0])
        if self.bc.y_max_type == 'mur':
            ez[1:-1, -1] = ez_prev[1:-1, -2] + coeff_y * (ez[1:-1, -2] - ez_prev[1:-1, -1])

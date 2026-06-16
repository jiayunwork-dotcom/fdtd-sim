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
    pml_layers: int = 12

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

        if self.has_pml:
            self._init_pml_coefficients()

    def _sigma_max(self, dx_or_dy):
        m = 4
        R0 = 1e-7
        return 12.0 * (m + 1) / (ETA0 * dx_or_dy)

    def _sigma_profile(self, depth, thickness, dx_or_dy):
        m = 4
        sigma_max = self._sigma_max(dx_or_dy)
        ratio = depth / thickness
        return sigma_max * (ratio ** m)

    def _kappa_profile(self, depth, thickness):
        m = 4
        kappa_max = 10.0
        ratio = depth / thickness
        return 1.0 + (kappa_max - 1.0) * (ratio ** m)

    def _alpha_profile(self, depth, thickness):
        alpha_max = 0.0
        return alpha_max

    def _init_pml_coefficients(self):
        nx, ny = self.nx, self.ny
        dx, dy, dt = self.dx, self.dy, self.dt

        sigma_e_x = np.zeros((nx, ny))
        sigma_h_x = np.zeros((nx, ny))
        kappa_e_x = np.ones((nx, ny))
        kappa_h_x = np.ones((nx, ny))
        alpha_e_x = np.zeros((nx, ny))
        alpha_h_x = np.zeros((nx, ny))

        sigma_e_y = np.zeros((nx, ny))
        sigma_h_y = np.zeros((nx, ny))
        kappa_e_y = np.ones((nx, ny))
        kappa_h_y = np.ones((nx, ny))
        alpha_e_y = np.zeros((nx, ny))
        alpha_h_y = np.zeros((nx, ny))

        if self.x_min_pml > 0:
            for i in range(self.x_min_pml):
                depth_e = self.x_min_pml - i
                depth_h = self.x_min_pml - i - 0.5
                if depth_h < 0:
                    depth_h = 0
                s_e = self._sigma_profile(depth_e, self.x_min_pml, dx)
                s_h = self._sigma_profile(depth_h, self.x_min_pml, dx)
                k_e = self._kappa_profile(depth_e, self.x_min_pml)
                k_h = self._kappa_profile(depth_h, self.x_min_pml)
                a_e = self._alpha_profile(depth_e, self.x_min_pml)
                a_h = self._alpha_profile(depth_h, self.x_min_pml)
                sigma_e_x[i, :] = s_e
                sigma_h_x[i, :] = s_h
                kappa_e_x[i, :] = k_e
                kappa_h_x[i, :] = k_h
                alpha_e_x[i, :] = a_e
                alpha_h_x[i, :] = a_h

        if self.x_max_pml > 0:
            for i in range(self.x_max_pml):
                depth_e = i + 1
                depth_h = i + 0.5
                s_e = self._sigma_profile(depth_e, self.x_max_pml, dx)
                s_h = self._sigma_profile(depth_h, self.x_max_pml, dx)
                k_e = self._kappa_profile(depth_e, self.x_max_pml)
                k_h = self._kappa_profile(depth_h, self.x_max_pml)
                a_e = self._alpha_profile(depth_e, self.x_max_pml)
                a_h = self._alpha_profile(depth_h, self.x_max_pml)
                idx = nx - self.x_max_pml + i
                sigma_e_x[idx, :] = s_e
                sigma_h_x[idx, :] = s_h
                kappa_e_x[idx, :] = k_e
                kappa_h_x[idx, :] = k_h
                alpha_e_x[idx, :] = a_e
                alpha_h_x[idx, :] = a_h

        if self.y_min_pml > 0:
            for j in range(self.y_min_pml):
                depth_e = self.y_min_pml - j
                depth_h = self.y_min_pml - j - 0.5
                if depth_h < 0:
                    depth_h = 0
                s_e = self._sigma_profile(depth_e, self.y_min_pml, dy)
                s_h = self._sigma_profile(depth_h, self.y_min_pml, dy)
                k_e = self._kappa_profile(depth_e, self.y_min_pml)
                k_h = self._kappa_profile(depth_h, self.y_min_pml)
                a_e = self._alpha_profile(depth_e, self.y_min_pml)
                a_h = self._alpha_profile(depth_h, self.y_min_pml)
                sigma_e_y[:, j] = s_e
                sigma_h_y[:, j] = s_h
                kappa_e_y[:, j] = k_e
                kappa_h_y[:, j] = k_h
                alpha_e_y[:, j] = a_e
                alpha_h_y[:, j] = a_h

        if self.y_max_pml > 0:
            for j in range(self.y_max_pml):
                depth_e = j + 1
                depth_h = j + 0.5
                s_e = self._sigma_profile(depth_e, self.y_max_pml, dy)
                s_h = self._sigma_profile(depth_h, self.y_max_pml, dy)
                k_e = self._kappa_profile(depth_e, self.y_max_pml)
                k_h = self._kappa_profile(depth_h, self.y_max_pml)
                a_e = self._alpha_profile(depth_e, self.y_max_pml)
                a_h = self._alpha_profile(depth_h, self.y_max_pml)
                jdx = ny - self.y_max_pml + j
                sigma_e_y[:, jdx] = s_e
                sigma_h_y[:, jdx] = s_h
                kappa_e_y[:, jdx] = k_e
                kappa_h_y[:, jdx] = k_h
                alpha_e_y[:, jdx] = a_e
                alpha_h_y[:, jdx] = a_h

        self.be_x = np.exp(-(sigma_e_x / kappa_e_x + alpha_e_x) * dt / EPS0)
        denom_e_x = sigma_e_x * kappa_e_x + kappa_e_x ** 2 * alpha_e_x
        safe_denom_e_x = np.where(denom_e_x > 1e-20, denom_e_x, 1.0)
        self.ce_x = np.where(denom_e_x > 1e-20,
                             sigma_e_x * (self.be_x - 1.0) / safe_denom_e_x, 0.0)

        self.be_y = np.exp(-(sigma_e_y / kappa_e_y + alpha_e_y) * dt / EPS0)
        denom_e_y = sigma_e_y * kappa_e_y + kappa_e_y ** 2 * alpha_e_y
        safe_denom_e_y = np.where(denom_e_y > 1e-20, denom_e_y, 1.0)
        self.ce_y = np.where(denom_e_y > 1e-20,
                             sigma_e_y * (self.be_y - 1.0) / safe_denom_e_y, 0.0)

        self.bh_x = np.exp(-(sigma_h_x / kappa_h_x + alpha_h_x) * dt / MU0)
        denom_h_x = sigma_h_x * kappa_h_x + kappa_h_x ** 2 * alpha_h_x
        safe_denom_h_x = np.where(denom_h_x > 1e-20, denom_h_x, 1.0)
        self.ch_x = np.where(denom_h_x > 1e-20,
                             sigma_h_x * (self.bh_x - 1.0) / safe_denom_h_x, 0.0)

        self.bh_y = np.exp(-(sigma_h_y / kappa_h_y + alpha_h_y) * dt / MU0)
        denom_h_y = sigma_h_y * kappa_h_y + kappa_h_y ** 2 * alpha_h_y
        safe_denom_h_y = np.where(denom_h_y > 1e-20, denom_h_y, 1.0)
        self.ch_y = np.where(denom_h_y > 1e-20,
                             sigma_h_y * (self.bh_y - 1.0) / safe_denom_h_y, 0.0)

        self.psi_e_x = np.zeros((nx, ny))
        self.psi_e_y = np.zeros((nx, ny))
        self.psi_h_x = np.zeros((nx, ny))
        self.psi_h_y = np.zeros((nx, ny))

    def update_psi_from_e(self, ez: np.ndarray):
        if not self.has_pml:
            return

        de_z_dx = np.zeros_like(ez)
        de_z_dx[1:, :] = (ez[1:, :] - ez[:-1, :]) / self.dx

        de_z_dy = np.zeros_like(ez)
        de_z_dy[:, 1:] = (ez[:, 1:] - ez[:, :-1]) / self.dy

        self.psi_h_x = self.bh_x * self.psi_h_x + self.ch_x * de_z_dx
        self.psi_h_y = self.bh_y * self.psi_h_y + self.ch_y * de_z_dy

    def update_psi_from_h(self, hx: np.ndarray, hy: np.ndarray):
        if not self.has_pml:
            return

        dh_y_dx = np.zeros_like(hy)
        dh_y_dx[1:, :] = (hy[1:, :] - hy[:-1, :]) / self.dx

        dh_x_dy = np.zeros_like(hx)
        dh_x_dy[:, 1:] = (hx[:, 1:] - hx[:, :-1]) / self.dy

        self.psi_e_x = self.be_x * self.psi_e_x + self.ce_x * dh_y_dx
        self.psi_e_y = self.be_y * self.psi_e_y + self.ce_y * dh_x_dy

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

import numpy as np
from typing import Tuple, Optional
import json

EPS0 = 8.854e-12
MU0 = 4 * np.pi * 1e-7
C0 = 1 / np.sqrt(EPS0 * MU0)
ETA0 = np.sqrt(MU0 / EPS0)


class NearFarFieldTransform:
    def __init__(self, near_field_data: dict, dx: float, dy: float, dt: float,
                 box_coords: Tuple[int, int, int, int], frequency: float):
        self.near_field_data = near_field_data
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.x1, self.y1, self.x2, self.y2 = box_coords
        self.frequency = frequency
        self.omega = 2 * np.pi * frequency
        self.k0 = self.omega / C0

    def _compute_equivalent_currents(self):
        nf = self.near_field_data

        ez_top = nf['ez_top']
        ez_bottom = nf['ez_bottom']
        ez_left = nf['ez_left']
        ez_right = nf['ez_right']

        hx_top = nf['hx_top']
        hx_bottom = nf['hx_bottom']
        hx_left = nf['hx_left']
        hx_right = nf['hx_right']

        hy_top = nf['hy_top']
        hy_bottom = nf['hy_bottom']
        hy_left = nf['hy_left']
        hy_right = nf['hy_right']

        nt = ez_top.shape[0]

        t = np.arange(nt) * self.dt
        freq_axis = np.fft.fftfreq(nt, self.dt)
        freq_idx = np.argmin(np.abs(freq_axis - self.frequency))

        def get_fft(data):
            return np.fft.fft(data, axis=0)[freq_idx]

        ez_top_f = get_fft(ez_top)
        ez_bottom_f = get_fft(ez_bottom)
        ez_left_f = get_fft(ez_left)
        ez_right_f = get_fft(ez_right)

        hx_top_f = get_fft(hx_top)
        hx_bottom_f = get_fft(hx_bottom)
        hx_left_f = get_fft(hx_left)
        hx_right_f = get_fft(hx_right)

        hy_top_f = get_fft(hy_top)
        hy_bottom_f = get_fft(hy_bottom)
        hy_left_f = get_fft(hy_left)
        hy_right_f = get_fft(hy_right)

        mx_top = -hy_top_f
        my_top = hx_top_f
        jx_top = np.zeros_like(ez_top_f)
        jy_top = ez_top_f

        mx_bottom = hy_bottom_f
        my_bottom = -hx_bottom_f
        jx_bottom = np.zeros_like(ez_bottom_f)
        jy_bottom = -ez_bottom_f

        mx_left = -hy_left_f
        my_left = hx_left_f
        jx_left = -ez_left_f
        jy_left = np.zeros_like(ez_left_f)

        mx_right = hy_right_f
        my_right = -hx_right_f
        jx_right = ez_right_f
        jy_right = np.zeros_like(ez_right_f)

        return {
            'top': (mx_top, my_top, jx_top, jy_top),
            'bottom': (mx_bottom, my_bottom, jx_bottom, jy_bottom),
            'left': (mx_left, my_left, jx_left, jy_left),
            'right': (mx_right, my_right, jx_right, jy_right)
        }

    def compute_far_field(self, num_angles: int = 360) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        currents = self._compute_equivalent_currents()

        angles = np.linspace(0, 2 * np.pi, num_angles)
        e_phi = np.zeros(num_angles, dtype=np.complex128)

        x1_m = self.x1 * self.dx
        y1_m = self.y1 * self.dy
        x2_m = self.x2 * self.dx
        y2_m = self.y2 * self.dy

        mx_top, my_top, jx_top, jy_top = currents['top']
        nx_top = len(mx_top)
        x_top = (self.x1 + np.arange(nx_top)) * self.dx
        y_top = y2_m

        mx_bottom, my_bottom, jx_bottom, jy_bottom = currents['bottom']
        nx_bottom = len(mx_bottom)
        x_bottom = (self.x1 + np.arange(nx_bottom)) * self.dx
        y_bottom = y1_m

        mx_left, my_left, jx_left, jy_left = currents['left']
        ny_left = len(mx_left)
        x_left = x1_m
        y_left = (self.y1 + np.arange(ny_left)) * self.dy

        mx_right, my_right, jx_right, jy_right = currents['right']
        ny_right = len(mx_right)
        x_right = x2_m
        y_right = (self.y1 + np.arange(ny_right)) * self.dy

        for idx, phi in enumerate(angles):
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            phase_top = self.k0 * (x_top * cos_phi + y_top * sin_phi)
            contrib_top = np.sum((jx_top * ETA0 + my_top * sin_phi - mx_top * cos_phi)
                                 * np.exp(-1j * phase_top)) * self.dx

            phase_bottom = self.k0 * (x_bottom * cos_phi + y_bottom * sin_phi)
            contrib_bottom = np.sum((jx_bottom * ETA0 + my_bottom * sin_phi - mx_bottom * cos_phi)
                                    * np.exp(-1j * phase_bottom)) * self.dx

            phase_left = self.k0 * (x_left * cos_phi + y_left * sin_phi)
            contrib_left = np.sum((jy_left * ETA0 - mx_left * sin_phi + my_left * cos_phi)
                                  * np.exp(-1j * phase_left)) * self.dy

            phase_right = self.k0 * (x_right * cos_phi + y_right * sin_phi)
            contrib_right = np.sum((jy_right * ETA0 - mx_right * sin_phi + my_right * cos_phi)
                                   * np.exp(-1j * phase_right)) * self.dy

            total = contrib_top + contrib_bottom + contrib_left + contrib_right
            e_phi[idx] = -1j * self.k0 * total / (4 * np.pi)

        far_field_magnitude = np.abs(e_phi)
        rcs = 2 * np.pi * np.abs(e_phi) ** 2 / (ETA0 * 1.0)
        rcs_db = 10 * np.log10(rcs + 1e-30)

        return angles, far_field_magnitude, rcs_db

    def to_dict(self):
        return {
            'dx': self.dx,
            'dy': self.dy,
            'dt': self.dt,
            'box_coords': [self.x1, self.y1, self.x2, self.y2],
            'frequency': self.frequency
        }

    @classmethod
    def from_dict(cls, data, near_field_data):
        return cls(
            near_field_data=near_field_data,
            dx=data['dx'],
            dy=data['dy'],
            dt=data['dt'],
            box_coords=tuple(data['box_coords']),
            frequency=data['frequency']
        )

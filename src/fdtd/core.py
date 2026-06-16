import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
import time
import json

from .materials import MaterialLibrary, StructureManager
from .sources import SourceManager
from .boundaries import BoundaryCondition, CPML

EPS0 = 8.854e-12
MU0 = 4 * np.pi * 1e-7
C0 = 1 / np.sqrt(EPS0 * MU0)
ETA0 = np.sqrt(MU0 / EPS0)


@dataclass
class SimulationConfig:
    width: float = 100e-6
    height: float = 100e-6
    dx: float = 1e-6
    dy: float = 1e-6
    total_time_steps: int = 1000
    dt: Optional[float] = None
    unit: str = 'um'
    sample_interval: int = 5
    observation_points: List[Tuple[int, int]] = field(default_factory=list)
    near_field_box: Optional[Tuple[int, int, int, int]] = None

    @property
    def nx(self) -> int:
        return int(self.width / self.dx)

    @property
    def ny(self) -> int:
        return int(self.height / self.dy)

    def courant_number(self) -> float:
        if self.dt is None:
            return 0
        return C0 * self.dt * np.sqrt(1 / self.dx ** 2 + 1 / self.dy ** 2)

    def max_stable_dt(self) -> float:
        return self.dx / (C0 * np.sqrt(2))

    def is_stable(self) -> bool:
        if self.dt is None:
            return True
        return self.courant_number() <= 1.0

    def to_dict(self):
        return {
            'width': self.width,
            'height': self.height,
            'dx': self.dx,
            'dy': self.dy,
            'total_time_steps': self.total_time_steps,
            'dt': self.dt,
            'unit': self.unit,
            'sample_interval': self.sample_interval,
            'observation_points': [list(p) for p in self.observation_points],
            'near_field_box': list(self.near_field_box) if self.near_field_box else None
        }

    @classmethod
    def from_dict(cls, data):
        cfg = cls()
        cfg.width = data.get('width', 100e-6)
        cfg.height = data.get('height', 100e-6)
        cfg.dx = data.get('dx', 1e-6)
        cfg.dy = data.get('dy', 1e-6)
        cfg.total_time_steps = data.get('total_time_steps', 1000)
        cfg.dt = data.get('dt')
        cfg.unit = data.get('unit', 'um')
        cfg.sample_interval = data.get('sample_interval', 5)
        cfg.observation_points = [tuple(p) for p in data.get('observation_points', [])]
        cfg.near_field_box = tuple(data['near_field_box']) if data.get('near_field_box') else None
        return cfg


@dataclass
class SimulationResult:
    ez_frames: np.ndarray
    hx_frames: np.ndarray
    hy_frames: np.ndarray
    ez_final: np.ndarray
    hx_final: np.ndarray
    hy_final: np.ndarray
    time_points: np.ndarray
    observation_times: np.ndarray
    observation_data: dict
    energy_density: np.ndarray
    near_field_data: Optional[dict] = None
    computation_time: float = 0.0

    def to_dict(self):
        return {
            'computation_time': self.computation_time,
            'time_points': self.time_points.tolist(),
            'observation_times': self.observation_times.tolist(),
            'observation_data': {str(k): v.tolist() for k, v in self.observation_data.items()},
            'energy_density': self.energy_density.tolist()
        }


class FDTD2D:
    def __init__(self, config: SimulationConfig,
                 material_lib: MaterialLibrary,
                 structure_mgr: StructureManager,
                 source_mgr: SourceManager,
                 boundary: BoundaryCondition):
        self.config = config
        self.material_lib = material_lib
        self.structure_mgr = structure_mgr
        self.source_mgr = source_mgr
        self.boundary = boundary

        x_min_pml, x_max_pml, y_min_pml, y_max_pml = boundary.get_pml_sizes()
        self.x_min_pml = x_min_pml
        self.x_max_pml = x_max_pml
        self.y_min_pml = y_min_pml
        self.y_max_pml = y_max_pml

        self.nx_total = config.nx + x_min_pml + x_max_pml
        self.ny_total = config.ny + y_min_pml + y_max_pml

        if config.dt is None:
            self.dt = config.max_stable_dt() * 0.9
        else:
            self.dt = config.dt

        self.dx = config.dx
        self.dy = config.dy

        self._init_fields()
        self._init_materials()
        self._init_coefficients()

        self.cpml = CPML(self.nx_total, self.ny_total, self.dx, self.dy, self.dt, boundary)

        self.ez_frames = []
        self.hx_frames = []
        self.hy_frames = []
        self.time_points = []
        self.observation_data = {}
        for pt in config.observation_points:
            self.observation_data[pt] = []

        self.near_field_data = None
        if config.near_field_box is not None:
            self.near_field_data = {
                'ez_top': [],
                'ez_bottom': [],
                'ez_left': [],
                'ez_right': [],
                'hx_top': [],
                'hx_bottom': [],
                'hx_left': [],
                'hx_right': [],
                'hy_top': [],
                'hy_bottom': [],
                'hy_left': [],
                'hy_right': []
            }

        source_mgr.init_tfsf(self.nx_total, self.ny_total, config.total_time_steps,
                             self.dx, self.dy, self.dt)

    def _init_fields(self):
        self.ez = np.zeros((self.nx_total, self.ny_total))
        self.hx = np.zeros((self.nx_total, self.ny_total))
        self.hy = np.zeros((self.nx_total, self.ny_total))
        self.ez_prev = np.zeros((self.nx_total, self.ny_total))

    def _init_materials(self):
        epsilon_r, sigma, mu_r, color_grid = self.structure_mgr.generate_material_grid(
            self.config.nx, self.config.ny, self.dx, self.dy, self.material_lib
        )

        self.epsilon_r = np.ones((self.nx_total, self.ny_total))
        self.sigma = np.zeros((self.nx_total, self.ny_total))
        self.mu_r = np.ones((self.nx_total, self.ny_total))
        self.color_grid = np.zeros((self.nx_total, self.ny_total, 3), dtype=np.uint8)

        self.epsilon_r[self.x_min_pml:self.x_min_pml + self.config.nx,
                       self.y_min_pml:self.y_min_pml + self.config.ny] = epsilon_r
        self.sigma[self.x_min_pml:self.x_min_pml + self.config.nx,
                   self.y_min_pml:self.y_min_pml + self.config.ny] = sigma
        self.mu_r[self.x_min_pml:self.x_min_pml + self.config.nx,
                  self.y_min_pml:self.y_min_pml + self.config.ny] = mu_r
        self.color_grid[self.x_min_pml:self.x_min_pml + self.config.nx,
                        self.y_min_pml:self.y_min_pml + self.config.ny] = color_grid

        if self.x_min_pml > 0 or self.x_max_pml > 0 or self.y_min_pml > 0 or self.y_max_pml > 0:
            self._apply_pml_conductivity()

    def _apply_pml_conductivity(self):
        m = 4
        sigma_max_e = 1.5 * (m + 1) / (ETA0 * self.dx)
        sigma_max_m = sigma_max_e * MU0 / EPS0

        x_indices = np.arange(self.nx_total)
        y_indices = np.arange(self.ny_total)
        X, Y = np.meshgrid(x_indices, y_indices, indexing='ij')

        dist = np.zeros_like(X, dtype=float)

        if self.x_min_pml > 0:
            d = (self.x_min_pml - X[:self.x_min_pml, :]) / self.x_min_pml
            dist[:self.x_min_pml, :] = np.maximum(dist[:self.x_min_pml, :], d)
        if self.x_max_pml > 0:
            x_start = self.nx_total - self.x_max_pml
            d = (X[x_start:, :] - (x_start - 1)) / self.x_max_pml
            dist[x_start:, :] = np.maximum(dist[x_start:, :], d)
        if self.y_min_pml > 0:
            d = (self.y_min_pml - Y[:, :self.y_min_pml]) / self.y_min_pml
            dist[:, :self.y_min_pml] = np.maximum(dist[:, :self.y_min_pml], d)
        if self.y_max_pml > 0:
            y_start = self.ny_total - self.y_max_pml
            d = (Y[:, y_start:] - (y_start - 1)) / self.y_max_pml
            dist[:, y_start:] = np.maximum(dist[:, y_start:], d)

        mask = dist > 0
        sigma_e = sigma_max_e * (dist[mask] ** m)
        sigma_m = sigma_max_m * (dist[mask] ** m)

        self.sigma[mask] = np.maximum(self.sigma[mask], sigma_e)

        self.sigma_m = np.zeros_like(self.mu_r)
        self.sigma_m[mask] = sigma_m

    def _init_coefficients(self):
        eps = EPS0 * self.epsilon_r
        mu = MU0 * self.mu_r

        self.ca = (1 - self.sigma * self.dt / (2 * eps)) / (1 + self.sigma * self.dt / (2 * eps))
        self.cb = (self.dt / eps) / (1 + self.sigma * self.dt / (2 * eps))

        if not hasattr(self, 'sigma_m'):
            self.sigma_m = np.zeros_like(self.mu_r)

        self.da = (1 - self.sigma_m * self.dt / (2 * mu)) / (1 + self.sigma_m * self.dt / (2 * mu))
        self.db = (self.dt / mu) / (1 + self.sigma_m * self.dt / (2 * mu))

    def _update_h(self, t_idx: int):
        ez = self.ez

        if self.x_min_pml > 0 or self.x_max_pml > 0 or self.y_min_pml > 0 or self.y_max_pml > 0:
            self.cpml.update_psi_from_e(ez)

        self.hx[:, :-1] = self.da[:, :-1] * self.hx[:, :-1] - \
                          self.db[:, :-1] * (ez[:, 1:] - ez[:, :-1]) / self.dy

        self.hy[:-1, :] = self.da[:-1, :] * self.hy[:-1, :] + \
                          self.db[:-1, :] * (ez[1:, :] - ez[:-1, :]) / self.dx

        self.source_mgr.apply_tfsf_h(self.ez, self.hx, self.hy, t_idx)

        if self.x_min_pml > 0 or self.x_max_pml > 0 or self.y_min_pml > 0 or self.y_max_pml > 0:
            self.cpml.apply_h(self.hx, self.hy, ez)

    def _update_e(self, t_idx: int, t: float):
        self.ez_prev[:, :] = self.ez

        hx = self.hx
        hy = self.hy

        if self.x_min_pml > 0 or self.x_max_pml > 0 or self.y_min_pml > 0 or self.y_max_pml > 0:
            self.cpml.update_psi_from_h(hx, hy)

        self.ez[1:-1, 1:-1] = self.ca[1:-1, 1:-1] * self.ez[1:-1, 1:-1] + \
                              self.cb[1:-1, 1:-1] * (
                                  (hy[1:-1, 1:-1] - hy[:-2, 1:-1]) / self.dx -
                                  (hx[1:-1, 1:-1] - hx[1:-1, :-2]) / self.dy
                              )

        self.source_mgr.apply_sources_e(self.ez, self.hx, self.hy, t, self.dt, self.dx, self.dy)
        self.source_mgr.apply_tfsf_e(self.ez, self.hx, self.hy, t_idx)

        self.cpml.apply_pec(self.ez, self.hx, self.hy)

        if self.x_min_pml > 0 or self.x_max_pml > 0 or self.y_min_pml > 0 or self.y_max_pml > 0:
            self.cpml.apply_e(self.ez, hx, hy)
        else:
            self.cpml.apply_mur(self.ez, self.ez_prev, self.dt, self.dx, self.dy)

        pec_mask = self.sigma > 1e19
        self.ez[pec_mask] = 0

    def _collect_near_field(self):
        if self.near_field_data is None or self.config.near_field_box is None:
            return

        x1, y1, x2, y2 = self.config.near_field_box
        x1 += self.x_min_pml
        x2 += self.x_min_pml
        y1 += self.y_min_pml
        y2 += self.y_min_pml

        self.near_field_data['ez_top'].append(self.ez[x1:x2 + 1, y2].copy())
        self.near_field_data['ez_bottom'].append(self.ez[x1:x2 + 1, y1].copy())
        self.near_field_data['ez_left'].append(self.ez[x1, y1:y2 + 1].copy())
        self.near_field_data['ez_right'].append(self.ez[x2, y1:y2 + 1].copy())

        self.near_field_data['hx_top'].append(self.hx[x1:x2 + 1, y2].copy())
        self.near_field_data['hx_bottom'].append(self.hx[x1:x2 + 1, y1].copy())
        self.near_field_data['hx_left'].append(self.hx[x1, y1:y2 + 1].copy())
        self.near_field_data['hx_right'].append(self.hx[x2, y1:y2 + 1].copy())

        self.near_field_data['hy_top'].append(self.hy[x1:x2 + 1, y2].copy())
        self.near_field_data['hy_bottom'].append(self.hy[x1:x2 + 1, y1].copy())
        self.near_field_data['hy_left'].append(self.hy[x1, y1:y2 + 1].copy())
        self.near_field_data['hy_right'].append(self.hy[x2, y1:y2 + 1].copy())

    def run(self, progress_callback: Optional[Callable[[int, int], None]] = None) -> SimulationResult:
        start_time = time.time()

        nt = self.config.total_time_steps
        sample_interval = self.config.sample_interval

        for t_idx in range(nt):
            t = t_idx * self.dt

            self._update_h(t_idx)
            self._update_e(t_idx, t)

            for pt in self.config.observation_points:
                i, j = pt
                gi, gj = i + self.x_min_pml, j + self.y_min_pml
                if 0 <= gi < self.nx_total and 0 <= gj < self.ny_total:
                    self.observation_data[pt].append(self.ez[gi, gj])

            if t_idx % sample_interval == 0:
                self.time_points.append(t)
                self.ez_frames.append(self._get_physical_domain(self.ez).copy())
                self.hx_frames.append(self._get_physical_domain(self.hx).copy())
                self.hy_frames.append(self._get_physical_domain(self.hy).copy())

            if self.near_field_data is not None:
                self._collect_near_field()

            if progress_callback is not None:
                progress_callback(t_idx + 1, nt)

        computation_time = time.time() - start_time

        ez_frames = np.array(self.ez_frames)
        hx_frames = np.array(self.hx_frames)
        hy_frames = np.array(self.hy_frames)
        time_points = np.array(self.time_points)
        observation_times = np.arange(nt) * self.dt

        observation_data_final = {}
        for pt, data in self.observation_data.items():
            observation_data_final[pt] = np.array(data)

        energy_density = 0.5 * (EPS0 * self.epsilon_r * self.ez ** 2 +
                                MU0 * self.mu_r * (self.hx ** 2 + self.hy ** 2))
        energy_density = self._get_physical_domain(energy_density)

        near_field_final = None
        if self.near_field_data is not None:
            near_field_final = {k: np.array(v) for k, v in self.near_field_data.items()}

        return SimulationResult(
            ez_frames=ez_frames,
            hx_frames=hx_frames,
            hy_frames=hy_frames,
            ez_final=self._get_physical_domain(self.ez),
            hx_final=self._get_physical_domain(self.hx),
            hy_final=self._get_physical_domain(self.hy),
            time_points=time_points,
            observation_times=observation_times,
            observation_data=observation_data_final,
            energy_density=energy_density,
            near_field_data=near_field_final,
            computation_time=computation_time
        )

    def _get_physical_domain(self, field: np.ndarray) -> np.ndarray:
        return field[self.x_min_pml:self.x_min_pml + self.config.nx,
                     self.y_min_pml:self.y_min_pml + self.config.ny]

    def get_color_grid_physical(self) -> np.ndarray:
        return self._get_physical_domain(self.color_grid)

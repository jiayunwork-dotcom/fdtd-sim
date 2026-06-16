import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import json

EPS0 = 8.854e-12
MU0 = 4 * np.pi * 1e-7
C0 = 1 / np.sqrt(EPS0 * MU0)


@dataclass
class Waveform:
    waveform_type: str
    frequency: float
    amplitude: float = 1.0
    bandwidth: float = 0.0
    start_time: float = 0.0

    def evaluate(self, t: float) -> float:
        if self.waveform_type == 'gaussian':
            tau = 0.5 / self.bandwidth
            t0 = 3 * tau
            return self.amplitude * np.exp(-((t - t0) ** 2) / (2 * tau ** 2))
        elif self.waveform_type == 'sine':
            omega = 2 * np.pi * self.frequency
            return self.amplitude * np.sin(omega * t)
        return 0.0

    def to_dict(self):
        return {
            'waveform_type': self.waveform_type,
            'frequency': self.frequency,
            'amplitude': self.amplitude,
            'bandwidth': self.bandwidth,
            'start_time': self.start_time
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            waveform_type=data['waveform_type'],
            frequency=data['frequency'],
            amplitude=data.get('amplitude', 1.0),
            bandwidth=data.get('bandwidth', 0.0),
            start_time=data.get('start_time', 0.0)
        )


@dataclass
class Source:
    source_type: str
    position: Tuple[int, int]
    waveform: Waveform
    direction: str = 'x'
    active: bool = True

    def to_dict(self):
        return {
            'source_type': self.source_type,
            'position': list(self.position),
            'waveform': self.waveform.to_dict(),
            'direction': self.direction,
            'active': self.active
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            source_type=data['source_type'],
            position=tuple(data['position']),
            waveform=Waveform.from_dict(data['waveform']),
            direction=data.get('direction', 'x'),
            active=data.get('active', True)
        )

    def apply(self, ez: np.ndarray, hx: np.ndarray, hy: np.ndarray,
              t: float, dt: float, dx: float, dy: float) -> None:
        if not self.active:
            return

        value = self.waveform.evaluate(t)

        if self.source_type == 'point':
            i, j = self.position
            if 0 <= i < ez.shape[0] and 0 <= j < ez.shape[1]:
                ez[i, j] += value

        elif self.source_type == 'line':
            i, j = self.position
            if self.direction == 'x':
                if 0 <= j < ez.shape[1]:
                    ez[:, j] += value
            elif self.direction == 'y':
                if 0 <= i < ez.shape[0]:
                    ez[i, :] += value


class TFSF:
    def __init__(self, x_min: int, x_max: int, y_min: int, y_max: int,
                 incident_angle: float, waveform: Waveform):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.incident_angle = incident_angle
        self.waveform = waveform
        self.nx = None
        self.ny = None
        self.nt = None
        self.dx = None
        self.dy = None
        self.dt = None
        self.theta = None
        self.kx = None
        self.ky = None
        self.omega = None

    def to_dict(self):
        return {
            'x_min': self.x_min,
            'x_max': self.x_max,
            'y_min': self.y_min,
            'y_max': self.y_max,
            'incident_angle': self.incident_angle,
            'waveform': self.waveform.to_dict()
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            x_min=data['x_min'],
            x_max=data['x_max'],
            y_min=data['y_min'],
            y_max=data['y_max'],
            incident_angle=data['incident_angle'],
            waveform=Waveform.from_dict(data['waveform'])
        )

    def init_incident(self, nx: int, ny: int, nt: int, dx: float, dy: float, dt: float):
        self.nx = nx
        self.ny = ny
        self.nt = nt
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.theta = np.radians(self.incident_angle)
        self.k0 = 2 * np.pi * self.waveform.frequency / C0
        self.kx = self.k0 * np.cos(self.theta)
        self.ky = self.k0 * np.sin(self.theta)
        self.omega = 2 * np.pi * self.waveform.frequency

    def _e_inc_at(self, i: int, j: int, t_idx: int) -> float:
        x = i * self.dx
        y = j * self.dy
        t = t_idx * self.dt
        phase = self.kx * x + self.ky * y - self.omega * t
        envelope = self.waveform.evaluate(t)
        return envelope * np.sin(phase)

    def _hx_inc_at(self, i: int, j: int, t_idx: int) -> float:
        return -np.sin(self.theta) / ETA0 * self._e_inc_at(i, j, t_idx)

    def _hy_inc_at(self, i: int, j: int, t_idx: int) -> float:
        return np.cos(self.theta) / ETA0 * self._e_inc_at(i, j, t_idx)

    def apply_e(self, ez: np.ndarray, hx: np.ndarray, hy: np.ndarray, t_idx: int):
        if self.theta is None:
            return

        nx, ny = ez.shape

        for i in range(self.x_min, self.x_max + 1):
            if 0 <= i < nx:
                if 0 <= self.y_min < ny:
                    ez[i, self.y_min] -= self._e_inc_at(i, self.y_min, t_idx)
                if 0 <= self.y_max < ny:
                    ez[i, self.y_max] -= self._e_inc_at(i, self.y_max, t_idx)

        for j in range(self.y_min, self.y_max + 1):
            if 0 <= j < ny:
                if 0 <= self.x_min < nx:
                    ez[self.x_min, j] -= self._e_inc_at(self.x_min, j, t_idx)
                if 0 <= self.x_max < nx:
                    ez[self.x_max, j] -= self._e_inc_at(self.x_max, j, t_idx)

    def apply_h(self, ez: np.ndarray, hx: np.ndarray, hy: np.ndarray, t_idx: int):
        if self.theta is None:
            return

        nx, ny = hx.shape
        hdt = t_idx + 0.5

        for i in range(self.x_min, self.x_max + 1):
            if 0 <= i < nx:
                if self.y_min - 1 >= 0 and self.y_min - 1 < ny:
                    hx[i, self.y_min - 1] += self._hx_inc_at(i, self.y_min - 1, hdt)
                if self.y_max >= 0 and self.y_max < ny:
                    hx[i, self.y_max] += self._hx_inc_at(i, self.y_max, hdt)

        for j in range(self.y_min, self.y_max + 1):
            if 0 <= j < ny:
                if self.x_min - 1 >= 0 and self.x_min - 1 < nx:
                    hy[self.x_min - 1, j] -= self._hy_inc_at(self.x_min - 1, j, hdt)
                if self.x_max >= 0 and self.x_max < nx:
                    hy[self.x_max, j] -= self._hy_inc_at(self.x_max, j, hdt)


class SourceManager:
    def __init__(self):
        self.sources: List[Source] = []
        self.tfsf: Optional[TFSF] = None

    def add_source(self, source: Source):
        self.sources.append(source)

    def remove_source(self, index: int):
        if 0 <= index < len(self.sources):
            self.sources.pop(index)

    def set_tfsf(self, tfsf: Optional[TFSF]):
        self.tfsf = tfsf

    def clear(self):
        self.sources = []
        self.tfsf = None

    def apply_sources_e(self, ez: np.ndarray, hx: np.ndarray, hy: np.ndarray,
                        t: float, dt: float, dx: float, dy: float):
        for source in self.sources:
            source.apply(ez, hx, hy, t, dt, dx, dy)

    def apply_sources_h(self, ez: np.ndarray, hx: np.ndarray, hy: np.ndarray,
                        t: float, dt: float, dx: float, dy: float):
        pass

    def apply_tfsf_e(self, ez: np.ndarray, hx: np.ndarray, hy: np.ndarray, t_idx: int):
        if self.tfsf:
            self.tfsf.apply_e(ez, hx, hy, t_idx)

    def apply_tfsf_h(self, ez: np.ndarray, hx: np.ndarray, hy: np.ndarray, t_idx: int):
        if self.tfsf:
            self.tfsf.apply_h(ez, hx, hy, t_idx)

    def init_tfsf(self, nx: int, ny: int, nt: int, dx: float, dy: float, dt: float):
        if self.tfsf:
            self.tfsf.init_incident(nx, ny, nt, dx, dy, dt)

    def to_dict(self):
        return {
            'sources': [s.to_dict() for s in self.sources],
            'tfsf': self.tfsf.to_dict() if self.tfsf else None
        }

    @classmethod
    def from_dict(cls, data):
        mgr = cls()
        mgr.sources = [Source.from_dict(s) for s in data.get('sources', [])]
        if data.get('tfsf'):
            mgr.tfsf = TFSF.from_dict(data['tfsf'])
        return mgr

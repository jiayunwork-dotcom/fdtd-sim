import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Dict
import copy
import warnings

from .core import FDTD2D, SimulationConfig, SimulationResult
from .materials import MaterialLibrary, StructureManager, Material, Structure
from .sources import SourceManager, Source, Waveform
from .boundaries import BoundaryCondition

EPS0 = 8.854e-12
MU0 = 4 * np.pi * 1e-7
C0 = 1 / np.sqrt(EPS0 * MU0)
ETA0 = np.sqrt(MU0 / EPS0)


@dataclass
class Port:
    port_index: int
    position: Tuple[int, int]
    direction: str
    z0: float = 50.0
    width: int = 3

    def __post_init__(self):
        valid_dirs = ['+x', '-x', '+y', '-y']
        if self.direction not in valid_dirs:
            raise ValueError(f"Direction must be one of {valid_dirs}")

    def get_sample_points(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        i, j = self.position
        if self.direction in ['+x', '-x']:
            return (i, j - 1), (i, j + 1)
        else:
            return (i - 1, j), (i + 1, j)

    def get_port_cross_section(self) -> List[Tuple[int, int]]:
        i, j = self.position
        points = []
        half_w = self.width // 2
        if self.direction in ['+x', '-x']:
            for dj in range(-half_w, half_w + 1):
                points.append((i, j + dj))
        else:
            for di in range(-half_w, half_w + 1):
                points.append((i + di, j))
        return points

    def get_current_loop(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        i, j = self.position
        half_w = self.width // 2
        hx_points = []
        hy_points = []
        if self.direction in ['+x', '-x']:
            for dj in range(-half_w, half_w + 1):
                hx_points.append((i, j + dj))
                hy_points.append((i - 1, j + dj))
                hy_points.append((i, j + dj))
        else:
            for di in range(-half_w, half_w + 1):
                hy_points.append((i + di, j))
                hx_points.append((i + di, j - 1))
                hx_points.append((i + di, j))
        return hx_points, hy_points

    def get_matched_load_sigma(self) -> float:
        return EPS0 * C0 * ETA0 / self.z0

    def to_dict(self):
        return {
            'port_index': self.port_index,
            'position': list(self.position),
            'direction': self.direction,
            'z0': self.z0,
            'width': self.width
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            port_index=data['port_index'],
            position=tuple(data['position']),
            direction=data['direction'],
            z0=data.get('z0', 50.0),
            width=data.get('width', 3)
        )


@dataclass
class SParameterConfig:
    center_frequency: float = 10e12
    bandwidth: float = 5e12
    amplitude: float = 1.0
    pulse_points: int = 200
    add_dc_bias: bool = False

    def to_dict(self):
        return {
            'center_frequency': self.center_frequency,
            'bandwidth': self.bandwidth,
            'amplitude': self.amplitude,
            'pulse_points': self.pulse_points,
            'add_dc_bias': self.add_dc_bias
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            center_frequency=data.get('center_frequency', 10e12),
            bandwidth=data.get('bandwidth', 5e12),
            amplitude=data.get('amplitude', 1.0),
            pulse_points=data.get('pulse_points', 200),
            add_dc_bias=data.get('add_dc_bias', False)
        )


@dataclass
class SParameterResult:
    frequencies: np.ndarray
    s_matrix: np.ndarray
    ports: List[Port]
    time_signals: Dict[str, Dict[int, Dict[str, np.ndarray]]]
    valid_bandwidth_mask: np.ndarray
    passivity_violation_mask: np.ndarray
    reciprocity_violation_mask: np.ndarray
    warnings: List[str] = field(default_factory=list)
    computation_time: float = 0.0

    @property
    def num_ports(self) -> int:
        return len(self.ports)

    def get_s_parameter(self, i: int, j: int) -> Tuple[np.ndarray, np.ndarray]:
        s_ij = self.s_matrix[i, j, :]
        mag_db = 20 * np.log10(np.abs(s_ij) + 1e-30)
        phase_deg = np.angle(s_ij, deg=True)
        return mag_db, phase_deg

    def check_signal_decay(self, threshold_db: float = -60.0) -> List[str]:
        warnings_list = []
        for excite_idx in range(self.num_ports):
            for port_idx in range(self.num_ports):
                key = f'excite_{excite_idx}'
                if key in self.time_signals:
                    port_data = self.time_signals[key].get(port_idx, {})
                    v_plus = port_data.get('v_plus', None)
                    v_minus = port_data.get('v_minus', None)
                    for sig_name, sig in [('V+', v_plus), ('V-', v_minus)]:
                        if sig is not None and len(sig) > 0:
                            peak = np.max(np.abs(sig))
                            if peak > 0:
                                final_val = np.max(np.abs(sig[-len(sig) // 10:]))
                                decay_db = 20 * np.log10(final_val / peak + 1e-30)
                                if decay_db > threshold_db:
                                    warnings_list.append(
                                        f"Port {port_idx + 1} {sig_name} signal only decayed to "
                                        f"{decay_db:.1f} dB (target {threshold_db} dB). "
                                        f"Consider increasing simulation time steps."
                                    )
        return warnings_list


class PortSampler:
    def __init__(self, port: Port, nx: int, ny: int, dt: float, dx: float, dy: float):
        self.port = port
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.v_plus: List[float] = []
        self.v_minus: List[float] = []
        self.i_total: List[float] = []
        self.v_total: List[float] = []

    def sample(self, ez: np.ndarray, hx: np.ndarray, hy: np.ndarray,
               x_min_pml: int, y_min_pml: int) -> None:
        v = self._compute_voltage(ez, x_min_pml, y_min_pml)
        i = self._compute_current(hx, hy, x_min_pml, y_min_pml)
        z0 = self.port.z0
        v_plus = (v + z0 * i) / 2
        v_minus = (v - z0 * i) / 2
        self.v_total.append(v)
        self.i_total.append(i)
        self.v_plus.append(v_plus)
        self.v_minus.append(v_minus)

    def _compute_voltage(self, ez: np.ndarray, x_min_pml: int, y_min_pml: int) -> float:
        cross_section = self.port.get_port_cross_section()
        voltages = []
        for (i, j) in cross_section:
            gi = i + x_min_pml
            gj = j + y_min_pml
            if 0 <= gi < ez.shape[0] and 0 <= gj < ez.shape[1]:
                voltages.append(ez[gi, gj] * self.dy)
        return np.mean(voltages) if voltages else 0.0

    def _compute_current(self, hx: np.ndarray, hy: np.ndarray,
                         x_min_pml: int, y_min_pml: int) -> float:
        i, j = self.port.position
        half_w = self.port.width // 2
        current = 0.0
        if self.port.direction in ['+x', '-x']:
            for dj in range(-half_w, half_w + 1):
                gj = j + dj + y_min_pml
                gi1 = i - 1 + x_min_pml
                gi2 = i + x_min_pml
                if 0 <= gj < hy.shape[1]:
                    if 0 <= gi1 < hy.shape[0]:
                        current += hy[gi1, gj] * self.dx
                    if 0 <= gi2 < hy.shape[0]:
                        current -= hy[gi2, gj] * self.dx
        else:
            for di in range(-half_w, half_w + 1):
                gi = i + di + x_min_pml
                gj1 = j - 1 + y_min_pml
                gj2 = j + y_min_pml
                if 0 <= gi < hx.shape[0]:
                    if 0 <= gj1 < hx.shape[1]:
                        current -= hx[gi, gj1] * self.dy
                    if 0 <= gj2 < hx.shape[1]:
                        current += hx[gi, gj2] * self.dy
        direction_sign = 1 if self.port.direction in ['+x', '+y'] else -1
        return direction_sign * current


class SParameterExtractor:
    def __init__(self,
                 config: SimulationConfig,
                 material_lib: MaterialLibrary,
                 structure_mgr: StructureManager,
                 boundary: BoundaryCondition,
                 ports: List[Port],
                 sparam_config: SParameterConfig):
        self.base_config = config
        self.base_material_lib = material_lib
        self.base_structure_mgr = structure_mgr
        self.base_boundary = boundary
        self.ports = ports
        self.sparam_config = sparam_config
        self.num_ports = len(ports)

        if self.num_ports < 2:
            raise ValueError("At least 2 ports are required for S-parameter extraction")

    def _create_gaussian_pulse_waveform(self) -> Waveform:
        return Waveform(
            waveform_type='gaussian',
            frequency=self.sparam_config.center_frequency,
            amplitude=self.sparam_config.amplitude,
            bandwidth=self.sparam_config.bandwidth
        )

    def _create_port_excitation(self, port_idx: int) -> Source:
        port = self.ports[port_idx]
        waveform = self._create_gaussian_pulse_waveform()
        return Source(
            source_type='line',
            position=port.position,
            waveform=waveform,
            direction='x' if port.direction in ['+x', '-x'] else 'y',
            active=True
        )

    def _add_matched_loads(self, material_lib: MaterialLibrary,
                           structure_mgr: StructureManager,
                           exclude_port_idx: int) -> None:
        for idx, port in enumerate(self.ports):
            if idx == exclude_port_idx:
                continue
            load_sigma = port.get_matched_load_sigma()
            load_mat = Material(
                name=f'MatchedLoad_Port{idx + 1}',
                epsilon_r=1.0,
                sigma=load_sigma,
                color='#888888'
            )
            material_lib.add_material(load_mat)
            i, j = port.position
            half_w = port.width // 2
            if port.direction in ['+x', '-x']:
                params = {
                    'x0': (i - 1) * self.base_config.dx,
                    'y0': (j - half_w) * self.base_config.dy,
                    'width': 3 * self.base_config.dx,
                    'height': (2 * half_w + 1) * self.base_config.dy
                }
            else:
                params = {
                    'x0': (i - half_w) * self.base_config.dx,
                    'y0': (j - 1) * self.base_config.dy,
                    'width': (2 * half_w + 1) * self.base_config.dx,
                    'height': 3 * self.base_config.dy
                }
            load_struct = Structure(
                shape_type='rectangle',
                material_name=load_mat.name,
                params=params,
                is_pec=False
            )
            structure_mgr.add_structure(load_struct)

    def _setup_simulation(self, excite_port_idx: int) -> Tuple[FDTD2D, List[PortSampler]]:
        config = copy.deepcopy(self.base_config)
        material_lib = copy.deepcopy(self.base_material_lib)
        structure_mgr = copy.deepcopy(self.base_structure_mgr)
        source_mgr = SourceManager()
        boundary = copy.deepcopy(self.base_boundary)
        self._add_matched_loads(material_lib, structure_mgr, excite_port_idx)
        excitation = self._create_port_excitation(excite_port_idx)
        source_mgr.add_source(excitation)
        fdtd = FDTD2D(config, material_lib, structure_mgr, source_mgr, boundary)
        samplers = []
        for port in self.ports:
            sampler = PortSampler(port, config.nx, config.ny, fdtd.dt, config.dx, config.dy)
            samplers.append(sampler)
        return fdtd, samplers

    def _run_single_simulation(self, excite_port_idx: int,
                               progress_callback: Optional[Callable[[int, int, int, int], None]] = None
                               ) -> Dict[int, Dict[str, np.ndarray]]:
        fdtd, samplers = self._setup_simulation(excite_port_idx)
        nt = self.base_config.total_time_steps
        sample_interval = self.base_config.sample_interval
        for t_idx in range(nt):
            t = t_idx * fdtd.dt
            fdtd._update_h(t_idx)
            fdtd._update_e(t_idx, t)
            for sampler in samplers:
                sampler.sample(fdtd.ez, fdtd.hx, fdtd.hy, fdtd.x_min_pml, fdtd.y_min_pml)
            if progress_callback is not None:
                progress_callback(excite_port_idx, self.num_ports, t_idx + 1, nt)
        results = {}
        for port_idx, sampler in enumerate(samplers):
            results[port_idx] = {
                'v_plus': np.array(sampler.v_plus),
                'v_minus': np.array(sampler.v_minus),
                'v_total': np.array(sampler.v_total),
                'i_total': np.array(sampler.i_total),
                'time': np.arange(nt) * fdtd.dt
            }
        return results

    def extract(self, progress_callback: Optional[Callable[[int, int, int, int], None]] = None
                ) -> SParameterResult:
        import time
        start_time = time.time()
        all_time_signals: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
        for excite_idx in range(self.num_ports):
            sim_results = self._run_single_simulation(excite_idx, progress_callback)
            all_time_signals[f'excite_{excite_idx}'] = sim_results
        frequencies, s_matrix, valid_mask = self._compute_sparameters(all_time_signals)
        passivity_mask = self._check_passivity(s_matrix)
        reciprocity_mask = self._check_reciprocity(s_matrix)
        computation_time = time.time() - start_time
        result = SParameterResult(
            frequencies=frequencies,
            s_matrix=s_matrix,
            ports=self.ports,
            time_signals=all_time_signals,
            valid_bandwidth_mask=valid_mask,
            passivity_violation_mask=passivity_mask,
            reciprocity_violation_mask=reciprocity_mask,
            computation_time=computation_time
        )
        result.warnings = result.check_signal_decay()
        return result

    def _compute_sparameters(self, time_signals: Dict[str, Dict[int, Dict[str, np.ndarray]]]
                             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ref_port_data = time_signals['excite_0'][0]
        v_plus_ref = ref_port_data['v_plus']
        time_arr = ref_port_data['time']
        nt = len(v_plus_ref)
        dt = time_arr[1] - time_arr[0] if len(time_arr) > 1 else 1e-15
        frequencies = np.fft.fftfreq(nt, dt)
        pos_mask = frequencies >= 0
        frequencies = frequencies[pos_mask]
        nf = len(frequencies)
        s_matrix = np.zeros((self.num_ports, self.num_ports, nf), dtype=complex)
        v_plus_incident = {}
        v_minus_scattered = {}
        for excite_idx in range(self.num_ports):
            excite_key = f'excite_{excite_idx}'
            v_plus_incident[excite_idx] = {}
            v_minus_scattered[excite_idx] = {}
            for port_idx in range(self.num_ports):
                port_data = time_signals[excite_key][port_idx]
                vp = port_data['v_plus']
                vm = port_data['v_minus']
                Vp_f = np.fft.fft(vp)[pos_mask]
                Vm_f = np.fft.fft(vm)[pos_mask]
                v_plus_incident[excite_idx][port_idx] = Vp_f
                v_minus_scattered[excite_idx][port_idx] = Vm_f
        z0_ref = self.ports[0].z0
        for i in range(self.num_ports):
            z0_i = self.ports[i].z0
            for j in range(self.num_ports):
                z0_j = self.ports[j].z0
                Vm_i = v_minus_scattered[j][i]
                Vp_j = v_plus_incident[j][j]
                denom = Vp_j + 1e-30
                s_ij = Vm_i / denom
                impedance_scale = np.sqrt(z0_i / z0_j) if z0_j > 0 else 1.0
                s_matrix[i, j, :] = s_ij * impedance_scale
        ref_vp = v_plus_incident[0][0]
        ref_mag = np.abs(ref_vp)
        peak_mag = np.max(ref_mag)
        if peak_mag > 0:
            bw_level = peak_mag * np.power(10, -20 / 20)
            valid_mask = ref_mag >= bw_level
        else:
            valid_mask = np.ones_like(frequencies, dtype=bool)
        return frequencies, s_matrix, valid_mask

    def _check_passivity(self, s_matrix: np.ndarray) -> np.ndarray:
        nf = s_matrix.shape[2]
        violation_mask = np.zeros(nf, dtype=bool)
        for f_idx in range(nf):
            S_f = s_matrix[:, :, f_idx]
            S_hermitian = S_f.conj().T
            product = S_hermitian @ S_f
            eigenvalues = np.linalg.eigvals(product)
            max_eig = np.max(np.real(eigenvalues))
            if max_eig > 1.0 + 1e-6:
                violation_mask[f_idx] = True
        return violation_mask

    def _check_reciprocity(self, s_matrix: np.ndarray, tolerance: float = 0.01) -> np.ndarray:
        nf = s_matrix.shape[2]
        violation_mask = np.zeros(nf, dtype=bool)
        for i in range(self.num_ports):
            for j in range(i + 1, self.num_ports):
                s_ij = s_matrix[i, j, :]
                s_ji = s_matrix[j, i, :]
                rel_diff = np.abs(s_ij - s_ji) / (np.maximum(np.abs(s_ij), np.abs(s_ji)) + 1e-30)
                violation_mask = violation_mask | (rel_diff > tolerance)
        return violation_mask


def compute_tdr(s_result: SParameterResult, port_idx: int = 0,
                impedance_ref: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    s11 = s_result.s_matrix[port_idx, port_idx, :]
    valid_mask = s_result.valid_bandwidth_mask
    s11_valid = s11.copy()
    s11_valid[~valid_mask] = 0.0
    nf = len(s_result.frequencies)
    frequencies = s_result.frequencies
    nt = 2 * (nf - 1)
    dt = 1 / (2 * frequencies[-1]) if frequencies[-1] > 0 else 1e-12
    s_full = np.zeros(nt, dtype=complex)
    s_full[:nf] = s11_valid
    if nf > 2:
        s_full[nf:] = np.conj(s11_valid[-2:0:-1])
    impulse_response = np.fft.ifft(s_full).real
    step_response = np.cumsum(impulse_response) * dt
    time_arr = np.arange(nt) * dt
    dist_arr = time_arr * C0 / 2
    step_response = np.clip(step_response, -0.99, 0.99)
    impedance = impedance_ref * (1 + step_response) / (1 - step_response)
    return dist_arr, impedance

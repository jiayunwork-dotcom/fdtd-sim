import numpy as np
from typing import Callable, Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

from .core import FDTD2D, SimulationConfig, SimulationResult
from .materials import MaterialLibrary, StructureManager
from .sources import SourceManager
from .boundaries import BoundaryCondition


@dataclass
class SweepConfig:
    param_name: str
    param_path: str
    start_value: float
    end_value: float
    num_steps: int
    metric: str
    metric_params: Dict[str, Any] = None

    @property
    def values(self) -> np.ndarray:
        return np.linspace(self.start_value, self.end_value, self.num_steps)


class ParametricSweep:
    def __init__(self, base_config: SimulationConfig,
                 base_material_lib: MaterialLibrary,
                 base_structure_mgr: StructureManager,
                 base_source_mgr: SourceManager,
                 base_boundary: BoundaryCondition):
        self.base_config = base_config
        self.base_material_lib = base_material_lib
        self.base_structure_mgr = base_structure_mgr
        self.base_source_mgr = base_source_mgr
        self.base_boundary = base_boundary

    def _apply_param(self, sweep_config: SweepConfig, value: float):
        import copy
        config = copy.deepcopy(self.base_config)
        mat_lib = copy.deepcopy(self.base_material_lib)
        struct_mgr = copy.deepcopy(self.base_structure_mgr)
        source_mgr = copy.deepcopy(self.base_source_mgr)
        boundary = copy.deepcopy(self.base_boundary)

        path = sweep_config.param_path
        parts = path.split('.')

        target = None
        if parts[0] == 'config':
            target = config
        elif parts[0] == 'material':
            mat_name = parts[1]
            if mat_name in mat_lib.materials:
                setattr(mat_lib.materials[mat_name], parts[2], value)
            return config, mat_lib, struct_mgr, source_mgr, boundary
        elif parts[0] == 'source':
            src_idx = int(parts[1])
            if src_idx < len(source_mgr.sources):
                src = source_mgr.sources[src_idx]
                if parts[2] == 'waveform':
                    setattr(src.waveform, parts[3], value)
                else:
                    setattr(src, parts[2], value)
            return config, mat_lib, struct_mgr, source_mgr, boundary
        elif parts[0] == 'tfsf':
            if source_mgr.tfsf:
                if parts[1] == 'waveform':
                    setattr(source_mgr.tfsf.waveform, parts[2], value)
                else:
                    setattr(source_mgr.tfsf, parts[1], value)
            return config, mat_lib, struct_mgr, source_mgr, boundary
        elif parts[0] == 'boundary':
            target = boundary

        if target is not None and len(parts) >= 2:
            setattr(target, parts[1], value)

        return config, mat_lib, struct_mgr, source_mgr, boundary

    def _compute_metric(self, result: SimulationResult, sweep_config: SweepConfig) -> float:
        metric = sweep_config.metric
        params = sweep_config.metric_params or {}

        if metric == 'peak_ez':
            point = params.get('point', (0, 0))
            point_key = tuple(point)
            if point_key in result.observation_data:
                return np.max(np.abs(result.observation_data[point_key]))
            return 0.0

        elif metric == 'peak_energy':
            return np.max(result.energy_density)

        elif metric == 'total_energy':
            return np.sum(result.energy_density)

        elif metric == 'transmittance':
            x1 = params.get('x1', 0)
            x2 = params.get('x2', result.ez_final.shape[0] - 1)
            y = params.get('y', result.ez_final.shape[1] - 1)
            incident_field = params.get('incident_field', 1.0)
            transmitted = np.mean(np.abs(result.ez_final[x1:x2 + 1, y]))
            return transmitted / incident_field if incident_field != 0 else 0.0

        elif metric == 'rcs_max':
            if result.near_field_data is not None:
                from .transforms import NearFarFieldTransform
                freq = params.get('frequency', 10e9)
                box = params.get('box', (0, 0, 10, 10))
                nf_transform = NearFarFieldTransform(
                    result.near_field_data,
                    self.base_config.dx, self.base_config.dy,
                    result.time_points[1] - result.time_points[0] if len(result.time_points) > 1 else 1e-12,
                    box, freq
                )
                angles, far_field, rcs_db = nf_transform.compute_far_field()
                return np.max(rcs_db)
            return 0.0

        return 0.0

    def run(self, sweep_config: SweepConfig,
            progress_callback: Optional[Callable[[int, int], None]] = None) -> Tuple[np.ndarray, np.ndarray, List[SimulationResult]]:
        param_values = sweep_config.values
        metric_values = np.zeros_like(param_values)
        results = []

        for i, value in enumerate(param_values):
            config, mat_lib, struct_mgr, source_mgr, boundary = self._apply_param(sweep_config, value)
            fdtd = FDTD2D(config, mat_lib, struct_mgr, source_mgr, boundary)
            result = fdtd.run()
            metric_values[i] = self._compute_metric(result, sweep_config)
            results.append(result)

            if progress_callback is not None:
                progress_callback(i + 1, len(param_values))

        return param_values, metric_values, results

import numpy as np
import json
import csv
import io
from typing import Dict, Any, Tuple

from .core import SimulationConfig, SimulationResult
from .materials import MaterialLibrary, StructureManager
from .sources import SourceManager
from .boundaries import BoundaryCondition


def export_config_json(config: SimulationConfig,
                       material_lib: MaterialLibrary,
                       structure_mgr: StructureManager,
                       source_mgr: SourceManager,
                       boundary: BoundaryCondition) -> str:
    data = {
        'config': config.to_dict(),
        'materials': material_lib.to_dict(),
        'structures': structure_mgr.to_dict(),
        'sources': source_mgr.to_dict(),
        'boundary': boundary.to_dict()
    }
    return json.dumps(data, indent=2)


def import_config_json(json_str: str) -> Tuple[SimulationConfig, MaterialLibrary, StructureManager, SourceManager, BoundaryCondition]:
    data = json.loads(json_str)

    config = SimulationConfig.from_dict(data.get('config', {}))
    mat_lib = MaterialLibrary.from_dict(data.get('materials', {}))
    struct_mgr = StructureManager.from_dict(data.get('structures', []))
    source_mgr = SourceManager.from_dict(data.get('sources', {}))
    boundary = BoundaryCondition.from_dict(data.get('boundary', {}))

    return config, mat_lib, struct_mgr, source_mgr, boundary


def export_field_csv(result: SimulationResult, field_type: str = 'ez') -> str:
    if field_type == 'ez':
        field = result.ez_final
    elif field_type == 'hx':
        field = result.hx_final
    elif field_type == 'hy':
        field = result.hy_final
    else:
        raise ValueError(f'Unknown field type: {field_type}')

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([f'{field_type.upper()} field (V/m or A/m)'])
    writer.writerow(['X index', 'Y index', 'Value'])

    nx, ny = field.shape
    for i in range(nx):
        for j in range(ny):
            writer.writerow([i, j, field[i, j]])

    return output.getvalue()


def export_all_fields_csv(result: SimulationResult) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['X index', 'Y index', 'Ez (V/m)', 'Hx (A/m)', 'Hy (A/m)'])

    nx, ny = result.ez_final.shape
    for i in range(nx):
        for j in range(ny):
            writer.writerow([
                i, j,
                result.ez_final[i, j],
                result.hx_final[i, j],
                result.hy_final[i, j]
            ])

    return output.getvalue()


def export_far_field_csv(angles: np.ndarray, far_field: np.ndarray,
                         rcs_db: np.ndarray) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Angle (rad)', 'Angle (deg)', 'Far Field Magnitude', 'RCS (dBsm)'])

    for ang, ff, rcs in zip(angles, far_field, rcs_db):
        writer.writerow([ang, np.degrees(ang), ff, rcs])

    return output.getvalue()


def export_observation_csv(result: SimulationResult) -> str:
    output = io.StringIO()
    writer = csv.writer(output)

    header = ['Time (s)', 'Time (ns)']
    for pt in result.observation_data.keys():
        header.append(f'Ez_({pt[0]},{pt[1]})')
    writer.writerow(header)

    nt = len(result.observation_times)
    for i in range(nt):
        row = [result.observation_times[i], result.observation_times[i] * 1e9]
        for pt, data in result.observation_data.items():
            if i < len(data):
                row.append(data[i])
            else:
                row.append('')
        writer.writerow(row)

    return output.getvalue()


def export_energy_csv(result: SimulationResult, dx: float, dy: float) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Energy Density (J/m^3)'])
    writer.writerow(['X index', 'Y index', 'Value', 'X (m)', 'Y (m)'])

    nx, ny = result.energy_density.shape
    for i in range(nx):
        for j in range(ny):
            writer.writerow([i, j, result.energy_density[i, j], i * dx, j * dy])

    return output.getvalue()


def save_config_to_file(config: SimulationConfig,
                        material_lib: MaterialLibrary,
                        structure_mgr: StructureManager,
                        source_mgr: SourceManager,
                        boundary: BoundaryCondition,
                        filepath: str):
    json_str = export_config_json(config, material_lib, structure_mgr, source_mgr, boundary)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(json_str)


def load_config_from_file(filepath: str) -> Tuple[SimulationConfig, MaterialLibrary, StructureManager, SourceManager, BoundaryCondition]:
    with open(filepath, 'r', encoding='utf-8') as f:
        json_str = f.read()
    return import_config_json(json_str)

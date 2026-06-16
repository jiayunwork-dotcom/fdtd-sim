import numpy as np
import json
import csv
import io
import datetime
from typing import Dict, Any, Tuple

from .core import SimulationConfig, SimulationResult
from .materials import MaterialLibrary, StructureManager
from .sources import SourceManager
from .boundaries import BoundaryCondition
from .sparam import SParameterResult, Port


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


def export_sparam_touchstone(s_result: SParameterResult,
                             parameter_format: str = 'MA',
                             frequency_unit: str = 'GHz',
                             valid_only: bool = True) -> str:
    valid_formats = ['MA', 'DB', 'RI']
    if parameter_format not in valid_formats:
        raise ValueError(f"Parameter format must be one of {valid_formats}")
    valid_freq_units = ['Hz', 'kHz', 'MHz', 'GHz', 'THz']
    if frequency_unit not in valid_freq_units:
        raise ValueError(f"Frequency unit must be one of {valid_freq_units}")
    freq_scale = {
        'Hz': 1.0,
        'kHz': 1e3,
        'MHz': 1e6,
        'GHz': 1e9,
        'THz': 1e12
    }[frequency_unit]
    num_ports = s_result.num_ports
    z0 = np.mean([p.z0 for p in s_result.ports])
    output = io.StringIO()
    output.write(f"! FDTD S-parameter Export\n")
    output.write(f"! Generated: {datetime.datetime.now().isoformat()}\n")
    output.write(f"! Number of ports: {num_ports}\n")
    for i, port in enumerate(s_result.ports):
        output.write(f"! Port {i+1}: position={port.position}, direction={port.direction}, Z0={port.z0} Ohm\n")
    output.write(f"! Valid bandwidth: {np.sum(s_result.valid_bandwidth_mask)}/{len(s_result.frequencies)} points\n")
    if np.any(s_result.passivity_violation_mask):
        output.write(f"! WARNING: {np.sum(s_result.passivity_violation_mask)} frequency points violate passivity\n")
    if np.any(s_result.reciprocity_violation_mask):
        output.write(f"! WARNING: {np.sum(s_result.reciprocity_violation_mask)} frequency points violate reciprocity\n")
    output.write(f"# {frequency_unit} S {parameter_format} R {z0:.1f}\n")
    mask = s_result.valid_bandwidth_mask if valid_only else np.ones_like(s_result.frequencies, dtype=bool)
    for f_idx in range(len(s_result.frequencies)):
        if not mask[f_idx]:
            continue
        freq = s_result.frequencies[f_idx] / freq_scale
        line = [f"{freq:.6e}"]
        for i in range(num_ports):
            for j in range(num_ports):
                s_ij = s_result.s_matrix[i, j, f_idx]
                if parameter_format == 'MA':
                    mag = np.abs(s_ij)
                    ang = np.angle(s_ij, deg=True)
                    line.append(f"{mag:.8e}")
                    line.append(f"{ang:.8e}")
                elif parameter_format == 'DB':
                    mag_db = 20 * np.log10(np.abs(s_ij) + 1e-30)
                    ang = np.angle(s_ij, deg=True)
                    line.append(f"{mag_db:.8e}")
                    line.append(f"{ang:.8e}")
                else:
                    line.append(f"{s_ij.real:.8e}")
                    line.append(f"{s_ij.imag:.8e}")
        output.write(" ".join(line) + "\n")
    return output.getvalue()


def export_sparam_csv(s_result: SParameterResult,
                      valid_only: bool = True) -> str:
    num_ports = s_result.num_ports
    output = io.StringIO()
    writer = csv.writer(output)
    header = ['Frequency (Hz)']
    for i in range(num_ports):
        for j in range(num_ports):
            header.append(f'S{i+1}{j+1}_real')
            header.append(f'S{i+1}{j+1}_imag')
            header.append(f'S{i+1}{j+1}_mag_dB')
            header.append(f'S{i+1}{j+1}_phase_deg')
    header.append('Valid_Bandwidth')
    header.append('Passivity_Violation')
    header.append('Reciprocity_Violation')
    writer.writerow(header)
    mask = s_result.valid_bandwidth_mask if valid_only else np.ones_like(s_result.frequencies, dtype=bool)
    for f_idx in range(len(s_result.frequencies)):
        if not mask[f_idx]:
            continue
        freq = s_result.frequencies[f_idx]
        row = [freq]
        for i in range(num_ports):
            for j in range(num_ports):
                s_ij = s_result.s_matrix[i, j, f_idx]
                mag_db = 20 * np.log10(np.abs(s_ij) + 1e-30)
                phase_deg = np.angle(s_ij, deg=True)
                row.append(s_ij.real)
                row.append(s_ij.imag)
                row.append(mag_db)
                row.append(phase_deg)
        row.append(1 if s_result.valid_bandwidth_mask[f_idx] else 0)
        row.append(1 if s_result.passivity_violation_mask[f_idx] else 0)
        row.append(1 if s_result.reciprocity_violation_mask[f_idx] else 0)
        writer.writerow(row)
    return output.getvalue()


def get_sparam_filename(num_ports: int, format: str = 'snp') -> str:
    if format == 'touchstone':
        return f'fdtd_sparam.{num_ports}p'
    else:
        return 'fdtd_sparam.csv'

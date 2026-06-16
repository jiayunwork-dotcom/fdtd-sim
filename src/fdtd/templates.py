from dataclasses import dataclass
from typing import Tuple

from .core import SimulationConfig
from .materials import MaterialLibrary, StructureManager, Structure
from .sources import SourceManager, Source, Waveform, TFSF
from .boundaries import BoundaryCondition


@dataclass
class Template:
    name: str
    description: str
    config: SimulationConfig
    material_lib: MaterialLibrary
    structure_mgr: StructureManager
    source_mgr: SourceManager
    boundary: BoundaryCondition
    observation_points: list
    near_field_box: Tuple[int, int, int, int] = None


def _base_config():
    return SimulationConfig(
        width=100e-6,
        height=100e-6,
        dx=1e-6,
        dy=1e-6,
        total_time_steps=500,
        unit='um',
        sample_interval=5
    )


def single_slit_diffraction() -> Template:
    config = SimulationConfig(
        width=200e-6,
        height=200e-6,
        dx=1e-6,
        dy=1e-6,
        total_time_steps=800,
        unit='um',
        sample_interval=5
    )

    mat_lib = MaterialLibrary()
    struct_mgr = StructureManager()

    slit_width = 20e-6
    barrier_thickness = 10e-6
    barrier_y = 80e-6

    struct_mgr.add_structure(Structure(
        shape_type='rectangle',
        material_name='PEC',
        params={'x0': 0, 'y0': barrier_y, 'width': 90e-6, 'height': barrier_thickness},
        is_pec=True
    ))
    struct_mgr.add_structure(Structure(
        shape_type='rectangle',
        material_name='PEC',
        params={'x0': 110e-6, 'y0': barrier_y, 'width': 90e-6, 'height': barrier_thickness},
        is_pec=True
    ))

    source_mgr = SourceManager()
    waveform = Waveform(
        waveform_type='gaussian',
        frequency=10e9,
        amplitude=1.0,
        bandwidth=5e9
    )
    tfsf = TFSF(
        x_min=10, x_max=190,
        y_min=10, y_max=190,
        incident_angle=270,
        waveform=waveform
    )
    source_mgr.set_tfsf(tfsf)

    boundary = BoundaryCondition(
        x_min_type='pml', x_max_type='pml',
        y_min_type='pml', y_max_type='pml',
        pml_layers=8
    )

    obs_points = [(100, 150), (50, 180), (150, 180)]

    return Template(
        name='单缝衍射',
        description='平面波入射到单缝上产生衍射图案',
        config=config,
        material_lib=mat_lib,
        structure_mgr=struct_mgr,
        source_mgr=source_mgr,
        boundary=boundary,
        observation_points=obs_points
    )


def double_slit_interference() -> Template:
    config = SimulationConfig(
        width=200e-6,
        height=200e-6,
        dx=1e-6,
        dy=1e-6,
        total_time_steps=800,
        unit='um',
        sample_interval=5
    )

    mat_lib = MaterialLibrary()
    struct_mgr = StructureManager()

    slit_width = 15e-6
    slit_sep = 40e-6
    barrier_thickness = 10e-6
    barrier_y = 80e-6
    center = 100e-6

    struct_mgr.add_structure(Structure(
        shape_type='rectangle',
        material_name='PEC',
        params={'x0': 0, 'y0': barrier_y, 'width': center - slit_sep / 2 - slit_width, 'height': barrier_thickness},
        is_pec=True
    ))
    struct_mgr.add_structure(Structure(
        shape_type='rectangle',
        material_name='PEC',
        params={'x0': center - slit_sep / 2, 'y0': barrier_y, 'width': slit_sep, 'height': barrier_thickness},
        is_pec=True
    ))
    struct_mgr.add_structure(Structure(
        shape_type='rectangle',
        material_name='PEC',
        params={'x0': center + slit_sep / 2 + slit_width, 'y0': barrier_y, 'width': 200e-6 - (center + slit_sep / 2 + slit_width), 'height': barrier_thickness},
        is_pec=True
    ))

    source_mgr = SourceManager()
    waveform = Waveform(
        waveform_type='gaussian',
        frequency=10e9,
        amplitude=1.0,
        bandwidth=5e9
    )
    tfsf = TFSF(
        x_min=10, x_max=190,
        y_min=10, y_max=190,
        incident_angle=270,
        waveform=waveform
    )
    source_mgr.set_tfsf(tfsf)

    boundary = BoundaryCondition(
        x_min_type='pml', x_max_type='pml',
        y_min_type='pml', y_max_type='pml',
        pml_layers=8
    )

    obs_points = [(100, 150), (60, 180), (140, 180)]

    return Template(
        name='双缝干涉',
        description='平面波入射到双缝上产生干涉条纹',
        config=config,
        material_lib=mat_lib,
        structure_mgr=struct_mgr,
        source_mgr=source_mgr,
        boundary=boundary,
        observation_points=obs_points
    )


def waveguide_transmission() -> Template:
    config = SimulationConfig(
        width=100e-6,
        height=200e-6,
        dx=1e-6,
        dy=1e-6,
        total_time_steps=800,
        unit='um',
        sample_interval=5
    )

    mat_lib = MaterialLibrary()
    struct_mgr = StructureManager()

    waveguide_width = 40e-6
    left_wall_x = 30e-6
    right_wall_x = 70e-6

    struct_mgr.add_structure(Structure(
        shape_type='rectangle',
        material_name='PEC',
        params={'x0': left_wall_x - 3e-6, 'y0': 0, 'width': 3e-6, 'height': 200e-6},
        is_pec=True
    ))
    struct_mgr.add_structure(Structure(
        shape_type='rectangle',
        material_name='PEC',
        params={'x0': right_wall_x, 'y0': 0, 'width': 3e-6, 'height': 200e-6},
        is_pec=True
    ))

    source_mgr = SourceManager()
    waveform = Waveform(
        waveform_type='sine',
        frequency=15e9,
        amplitude=1.0
    )
    source_mgr.add_source(Source(
        source_type='line',
        position=(50, 20),
        waveform=waveform,
        direction='x'
    ))

    boundary = BoundaryCondition(
        x_min_type='pml', x_max_type='pml',
        y_min_type='mur', y_max_type='mur',
        pml_layers=8
    )

    obs_points = [(50, 50), (50, 100), (50, 150)]

    return Template(
        name='波导传输',
        description='电磁波在平行板波导中的传输',
        config=config,
        material_lib=mat_lib,
        structure_mgr=struct_mgr,
        source_mgr=source_mgr,
        boundary=boundary,
        observation_points=obs_points
    )


def dielectric_sphere_scattering() -> Template:
    config = SimulationConfig(
        width=150e-6,
        height=150e-6,
        dx=1e-6,
        dy=1e-6,
        total_time_steps=600,
        unit='um',
        sample_interval=5
    )

    mat_lib = MaterialLibrary()
    struct_mgr = StructureManager()

    sphere_radius = 20e-6
    sphere_cx = 75e-6
    sphere_cy = 75e-6

    struct_mgr.add_structure(Structure(
        shape_type='circle',
        material_name='Glass',
        params={'cx': sphere_cx, 'cy': sphere_cy, 'radius': sphere_radius}
    ))

    source_mgr = SourceManager()
    waveform = Waveform(
        waveform_type='gaussian',
        frequency=10e9,
        amplitude=1.0,
        bandwidth=5e9
    )
    tfsf = TFSF(
        x_min=15, x_max=135,
        y_min=15, y_max=135,
        incident_angle=0,
        waveform=waveform
    )
    source_mgr.set_tfsf(tfsf)

    boundary = BoundaryCondition(
        x_min_type='pml', x_max_type='pml',
        y_min_type='pml', y_max_type='pml',
        pml_layers=10
    )

    obs_points = [(30, 75), (120, 75)]

    near_field_box = (25, 25, 125, 125)

    return Template(
        name='介质球散射',
        description='平面波入射到介质球上的散射，含近远场变换',
        config=config,
        material_lib=mat_lib,
        structure_mgr=struct_mgr,
        source_mgr=source_mgr,
        boundary=boundary,
        observation_points=obs_points,
        near_field_box=near_field_box
    )


def microstrip_line() -> Template:
    config = SimulationConfig(
        width=100e-6,
        height=60e-6,
        dx=0.5e-6,
        dy=0.5e-6,
        total_time_steps=600,
        unit='um',
        sample_interval=5
    )

    mat_lib = MaterialLibrary()
    struct_mgr = StructureManager()

    substrate_thickness = 10e-6
    substrate_y = 5e-6
    strip_width = 10e-6
    strip_x = 45e-6
    strip_y = substrate_y + substrate_thickness

    struct_mgr.add_structure(Structure(
        shape_type='rectangle',
        material_name='Silicon',
        params={'x0': 0, 'y0': substrate_y, 'width': 100e-6, 'height': substrate_thickness}
    ))

    struct_mgr.add_structure(Structure(
        shape_type='rectangle',
        material_name='PEC',
        params={'x0': 0, 'y0': 0, 'width': 100e-6, 'height': 2e-6},
        is_pec=True
    ))

    struct_mgr.add_structure(Structure(
        shape_type='rectangle',
        material_name='PEC',
        params={'x0': strip_x, 'y0': strip_y, 'width': strip_width, 'height': 1.5e-6},
        is_pec=True
    ))

    source_mgr = SourceManager()
    waveform = Waveform(
        waveform_type='gaussian',
        frequency=20e9,
        amplitude=1.0,
        bandwidth=10e9
    )
    source_mgr.add_source(Source(
        source_type='point',
        position=(100, int((strip_y + 1e-6) / 0.5e-6)),
        waveform=waveform
    ))

    boundary = BoundaryCondition(
        x_min_type='pml', x_max_type='pml',
        y_min_type='pml', y_max_type='pml',
        pml_layers=8
    )

    obs_points = [(50, int(strip_y / 0.5e-6)), (150, int(strip_y / 0.5e-6))]

    return Template(
        name='微带线',
        description='介质基板上的微带传输线，点源馈电',
        config=config,
        material_lib=mat_lib,
        structure_mgr=struct_mgr,
        source_mgr=source_mgr,
        boundary=boundary,
        observation_points=obs_points
    )


TEMPLATE_REGISTRY = {
    'single_slit': single_slit_diffraction,
    'double_slit': double_slit_interference,
    'waveguide': waveguide_transmission,
    'dielectric_sphere': dielectric_sphere_scattering,
    'microstrip': microstrip_line
}


def list_templates():
    templates = []
    for key, func in TEMPLATE_REGISTRY.items():
        tpl = func()
        templates.append({'key': key, 'name': tpl.name, 'description': tpl.description})
    return templates


def get_template(key: str) -> Template:
    if key not in TEMPLATE_REGISTRY:
        raise ValueError(f'Unknown template: {key}')
    return TEMPLATE_REGISTRY[key]()

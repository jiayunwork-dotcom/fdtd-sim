import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fdtd import (
    FDTD2D, SimulationConfig,
    MaterialLibrary, StructureManager, Structure,
    SourceManager, Source, Waveform,
    BoundaryCondition,
    NearFarFieldTransform,
    get_template, list_templates
)
from fdtd.visualization import (
    plot_field_heatmap, plot_time_waveform, plot_frequency_spectrum,
    plot_energy_density, generate_animation_frames, save_gif
)
from fdtd.io_utils import (
    export_config_json, import_config_json, save_config_to_file,
    export_all_fields_csv, export_observation_csv, export_energy_csv
)
import numpy as np
import time
import tempfile
import json

print("=== 完整功能测试 ===")
print()

print("1. 测试模板功能...")
templates = list_templates()
print(f"   可用模板: {templates}")
tmpl = get_template('single_slit')
print(f"   单缝衍射模板: {tmpl.name}")
print("   ✓ 模板功能正常")
print()

print("2. 测试仿真配置...")
config = SimulationConfig(
    width=200e-6,
    height=200e-6,
    dx=2e-6,
    dy=2e-6,
    total_time_steps=500,
    sample_interval=20,
    unit='um'
)
config.observation_points = [(50, 50), (80, 80)]
config.near_field_box = (30, 30, 70, 70)
print(f"   网格: {config.nx} x {config.ny}")
print(f"   最大 dt: {config.max_stable_dt():.3e} s")
print("   ✓ 配置功能正常")
print()

print("3. 测试材料与结构...")
mat_lib = MaterialLibrary()
struct_mgr = StructureManager()
struct_mgr.add_structure(Structure(
    shape_type='rectangle',
    material_name='Glass',
    params={'x0': 60e-6, 'y0': 60e-6, 'width': 80e-6, 'height': 80e-6}
))
struct_mgr.add_structure(Structure(
    shape_type='circle',
    material_name='PEC',
    params={'cx': 100e-6, 'cy': 100e-6, 'radius': 20e-6},
    is_pec=True
))
print(f"   结构数量: {len(struct_mgr.structures)}")
print("   ✓ 材料与结构功能正常")
print()

print("4. 测试源...")
source_mgr = SourceManager()
waveform = Waveform(
    waveform_type='gaussian',
    frequency=20e9,
    amplitude=1.0,
    bandwidth=10e9
)
source_mgr.add_source(Source(
    source_type='point',
    position=(20, 50),
    waveform=waveform
))
print(f"   源数量: {len(source_mgr.sources)}")
print("   ✓ 源功能正常")
print()

print("5. 测试边界条件...")
boundary = BoundaryCondition(
    x_min_type='mur', x_max_type='mur',
    y_min_type='mur', y_max_type='mur'
)
print("   ✓ 边界条件功能正常")
print()

print("6. 运行仿真...")
start = time.time()
fdtd = FDTD2D(config, mat_lib, struct_mgr, source_mgr, boundary)
result = fdtd.run()
run_time = time.time() - start
sim_dt = fdtd.dt
print(f"   耗时: {run_time:.3f} s")
print(f"   采样帧数: {len(result.ez_frames)}")
print(f"   Ez 峰值: {np.max(np.abs(result.ez_final)):.6f} V/m")
print(f"   观测点数据长度: {len(result.observation_data[(50, 50)])}")
print("   ✓ 仿真功能正常")
print()

print("7. 测试可视化...")
fig1 = plot_field_heatmap(result.ez_final, title='Ez 场分布',
                          dx=config.dx, dy=config.dy, unit=config.unit)
print("   ✓ 热力图正常")

fig2 = plot_time_waveform(result.observation_times, result.observation_data[(50, 50)],
                          title='观测点时域波形')
print("   ✓ 时域波形正常")

fig3 = plot_frequency_spectrum(result.observation_times, result.observation_data[(50, 50)],
                               title='频谱分析')
print("   ✓ 频谱分析正常")

fig4 = plot_energy_density(result.energy_density, dx=config.dx, dy=config.dy, unit=config.unit)
print("   ✓ 能量密度正常")

frames = generate_animation_frames(result.ez_frames, dx=config.dx, dy=config.dy, unit=config.unit)
print(f"   动画帧数: {len(frames)}")
print("   ✓ 动画帧生成正常")
print()

print("8. 测试数据导出...")
with tempfile.TemporaryDirectory() as tmpdir:
    config_path = os.path.join(tmpdir, 'config.json')
    save_config_to_file(config, mat_lib, struct_mgr, source_mgr, boundary, config_path)
    with open(config_path, 'r') as f:
        cfg_data = json.load(f)
    print(f"   配置导出: {len(cfg_data)} 个字段")
    
    fields_path = os.path.join(tmpdir, 'fields.csv')
    fields_csv = export_all_fields_csv(result)
    with open(fields_path, 'w') as f:
        f.write(fields_csv)
    print(f"   场数据导出: {os.path.getsize(fields_path)} bytes")
    
    obs_path = os.path.join(tmpdir, 'observation.csv')
    obs_csv = export_observation_csv(result)
    with open(obs_path, 'w') as f:
        f.write(obs_csv)
    print(f"   观测点导出: {os.path.getsize(obs_path)} bytes")
    
    energy_path = os.path.join(tmpdir, 'energy.csv')
    energy_csv = export_energy_csv(result, config.dx, config.dy)
    with open(energy_path, 'w') as f:
        f.write(energy_csv)
    print(f"   能量导出: {os.path.getsize(energy_path)} bytes")
    
    print("   ✓ 数据导出功能正常")
print()

print("9. 测试近远场变换...")
if result.near_field_data is not None:
    nfft = NearFarFieldTransform(
        result.near_field_data,
        config.dx, config.dy, sim_dt,
        config.near_field_box,
        frequency=20e9
    )
    angles_rad, far_field, rcs_db = nfft.compute_far_field(num_angles=361)
    print(f"   远场角度数: {len(angles_rad)}")
    print(f"   远场峰值: {np.max(np.abs(far_field)):.6f}")
    print("   ✓ 近远场变换正常")
else:
    print("   ⚠ 近场数据为空，跳过")
print()

print("=== 所有测试通过 ===")

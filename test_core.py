import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fdtd import (
    FDTD2D, SimulationConfig,
    MaterialLibrary, StructureManager, Structure,
    SourceManager, Source, Waveform,
    BoundaryCondition
)
import numpy as np
import time

print("=== FDTD 核心功能测试 ===")

config = SimulationConfig(
    width=1000e-6,
    height=1000e-6,
    dx=10e-6,
    dy=10e-6,
    total_time_steps=2000,
    unit='um',
    sample_interval=50
)

print(f"网格尺寸: {config.nx} x {config.ny}")
print(f"最大稳定时间步长: {config.max_stable_dt():.3e} s")
print(f"Courant数 (自动): {config.courant_number():.4f}")

mat_lib = MaterialLibrary()
struct_mgr = StructureManager()

struct_mgr.add_structure(Structure(
    shape_type='rectangle',
    material_name='Glass',
    params={'x0': 300e-6, 'y0': 300e-6, 'width': 400e-6, 'height': 400e-6}
))

struct_mgr.add_structure(Structure(
    shape_type='circle',
    material_name='PEC',
    params={'cx': 700e-6, 'cy': 700e-6, 'radius': 100e-6},
    is_pec=True
))

source_mgr = SourceManager()
waveform = Waveform(
    waveform_type='sine',
    frequency=5e9,
    amplitude=1.0
)
source_mgr.add_source(Source(
    source_type='point',
    position=(20, 50),
    waveform=waveform
))

boundary = BoundaryCondition(
    x_min_type='pml', x_max_type='pml',
    y_min_type='pml', y_max_type='pml',
    pml_layers=10
)

print("\n正在初始化 FDTD 仿真器...")
start = time.time()
fdtd = FDTD2D(config, mat_lib, struct_mgr, source_mgr, boundary)
init_time = time.time() - start
print(f"初始化完成，耗时: {init_time:.3f} s")

print(f"总网格 (含PML): {fdtd.nx_total} x {fdtd.ny_total}")

print("\n正在运行仿真...")
start = time.time()
result = fdtd.run()
run_time = time.time() - start
print(f"仿真完成，耗时: {run_time:.3f} s")
print(f"每步平均: {run_time/config.total_time_steps*1000:.3f} ms")

print(f"\n结果统计:")
print(f"  采样帧数: {len(result.ez_frames)}")
print(f"  Ez 峰值: {np.max(np.abs(result.ez_final)):.4f} V/m")
print(f"  Hx 峰值: {np.max(np.abs(result.hx_final)):.4f} A/m")
print(f"  能量密度峰值: {np.max(result.energy_density):.6e} J/m^3")

print(f"\n源位置(20,50)处的 Ez: {result.ez_final[20, 50]:.6f} V/m")
print(f"Ez 场最大位置: {np.unravel_index(np.argmax(np.abs(result.ez_final)), result.ez_final.shape)}")
print(f"Ez 最大值: {np.max(np.abs(result.ez_final)):.6f} V/m")

print("\n=== 测试通过 ===")

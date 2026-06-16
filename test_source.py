import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fdtd import (
    FDTD2D, SimulationConfig,
    MaterialLibrary, StructureManager,
    SourceManager, Source, Waveform,
    BoundaryCondition
)
import numpy as np

print("=== 单步源测试 ===")

config = SimulationConfig(
    width=1000e-6,
    height=1000e-6,
    dx=10e-6,
    dy=10e-6,
    total_time_steps=1,
    unit='um',
    sample_interval=1
)

print(f"dt = {config.max_stable_dt() * 0.9:.6e} s")

mat_lib = MaterialLibrary()
struct_mgr = StructureManager()

source_mgr = SourceManager()
waveform = Waveform(
    waveform_type='sine',
    frequency=5e9,
    amplitude=1.0
)
source_mgr.add_source(Source(
    source_type='point',
    position=(50, 50),
    waveform=waveform
))

boundary = BoundaryCondition(
    x_min_type='mur', x_max_type='mur',
    y_min_type='mur', y_max_type='mur',
    pml_layers=8
)

fdtd = FDTD2D(config, mat_lib, struct_mgr, source_mgr, boundary)

print(f"\n初始状态:")
print(f"  Ez[50,50] = {fdtd.ez[50, 50]}")
print(f"  Hx[50,50] = {fdtd.hx[50, 50]}")

t = 0
waveform_value = waveform.evaluate(t)
print(f"\n波形在 t={t:.2e} s 时的值: {waveform_value:.6f}")

print("\n运行 1 步...")
result = fdtd.run()

print(f"\n1 步后:")
print(f"  Ez[50,50] = {result.ez_final[50, 50]:.10f}")
print(f"  Ez 最大值 = {np.max(np.abs(result.ez_final)):.10f}")
print(f"  最大值位置 = {np.unravel_index(np.argmax(np.abs(result.ez_final)), result.ez_final.shape)}")

print("\n运行 2 步...")
config.total_time_steps = 2
fdtd2 = FDTD2D(config, mat_lib, struct_mgr, source_mgr, boundary)
result2 = fdtd2.run()
print(f"  Ez[50,50] = {result2.ez_final[50, 50]:.10f}")

print("\n运行 5 步...")
config.total_time_steps = 5
fdtd3 = FDTD2D(config, mat_lib, struct_mgr, source_mgr, boundary)
result3 = fdtd3.run()
print(f"  Ez[50,50] = {result3.ez_final[50, 50]:.10f}")

print("\n运行 10 步...")
config.total_time_steps = 10
fdtd4 = FDTD2D(config, mat_lib, struct_mgr, source_mgr, boundary)
result4 = fdtd4.run()
print(f"  Ez[50,50] = {result4.ez_final[50, 50]:.10f}")

print("\n=== 测试完成 ===")

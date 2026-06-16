import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import io
import sys
import os
import copy
import time

EPS0 = 8.854e-12
MU0 = 4 * np.pi * 1e-7
C0 = 1 / np.sqrt(EPS0 * MU0)
ETA0 = np.sqrt(MU0 / EPS0)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fdtd import (
    FDTD2D, SimulationConfig, SimulationResult,
    Material, MaterialLibrary, Structure, StructureManager,
    Source, Waveform, TFSF, SourceManager,
    BoundaryCondition, CPML,
    NearFarFieldTransform,
    list_templates, get_template
)
from fdtd.visualization import (
    plot_field_heatmap, plot_time_waveform, plot_frequency_spectrum,
    plot_far_field_polar, plot_rcs, plot_energy_density,
    fig_to_image, fig_to_bytes, generate_animation_frames, save_gif,
    plot_parametric_sweep
)
from fdtd.io_utils import (
    export_config_json, import_config_json,
    export_all_fields_csv, export_far_field_csv,
    export_observation_csv, export_energy_csv
)
from fdtd.parametric_sweep import ParametricSweep, SweepConfig

st.set_page_config(
    page_title='FDTD 电磁场仿真工具',
    page_icon='📡',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .main .block-container {
        padding-top: 1rem;
    }
    h1, h2, h3 {
        color: #1f4e79;
    }
    .stMetric {
        background-color: white;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def init_state():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.config = SimulationConfig(
            width=100e-6,
            height=100e-6,
            dx=1e-6,
            dy=1e-6,
            total_time_steps=500,
            unit='um',
            sample_interval=5
        )
        st.session_state.material_lib = MaterialLibrary()
        st.session_state.structure_mgr = StructureManager()
        st.session_state.source_mgr = SourceManager()
        st.session_state.boundary = BoundaryCondition()
        st.session_state.result = None
        st.session_state.far_field_result = None
        st.session_state.current_frame = 0
        st.session_state.is_playing = False
        st.session_state.observation_points = []
        st.session_state.near_field_box = None
        st.session_state.selected_obs_point = 0
        st.session_state.analysis_tab = 0

        st.session_state.compare_mode = False
        st.session_state.compare_material_lib = MaterialLibrary()
        st.session_state.compare_structure_mgr = StructureManager()
        st.session_state.compare_source_mgr = SourceManager()
        st.session_state.compare_boundary = BoundaryCondition()
        st.session_state.compare_result = None
        st.session_state.compare_color_grid = None

        st.session_state.bookmarks = []
        st.session_state.convergence_result = None


def unit_to_factor(unit):
    return 1e-6 if unit == 'um' else 1e-3


def factor_to_unit(unit):
    return 1e6 if unit == 'um' else 1e3


init_state()

with st.sidebar:
    st.title('⚙️ 仿真配置')

    compare_mode = st.toggle('🔄 对比模式', value=st.session_state.compare_mode, key='compare_mode_toggle')
    st.session_state.compare_mode = compare_mode

    if compare_mode:
        st.info('💡 对比模式已开启，左右面板共享域尺寸和时间步')

    with st.expander('📐 仿真域与网格', expanded=True):
        unit = st.selectbox('长度单位', ['um', 'mm'], index=0, key='unit_select')
        factor = factor_to_unit(unit)

        config = st.session_state.config
        config.unit = unit

        col1, col2 = st.columns(2)
        with col1:
            width = st.number_input('宽度', value=config.width * factor, min_value=1.0, max_value=10000.0,
                                    step=1.0, key='domain_width')
            config.width = width / factor
        with col2:
            height = st.number_input('高度', value=config.height * factor, min_value=1.0, max_value=10000.0,
                                     step=1.0, key='domain_height')
            config.height = height / factor

        dx = st.number_input('空间步长 dx=dy', value=config.dx * factor, min_value=0.01, max_value=100.0,
                             step=0.1, key='dx_input')
        config.dx = dx / factor
        config.dy = dx / factor

        total_steps = st.number_input('总时间步数', value=config.total_time_steps, min_value=10, max_value=100000,
                                      step=10, key='total_steps')
        config.total_time_steps = int(total_steps)

        sample_interval = st.number_input('采样间隔', value=config.sample_interval, min_value=1, max_value=100,
                                          step=1, key='sample_interval')
        config.sample_interval = int(sample_interval)

        nx = config.nx
        ny = config.ny
        max_dt = config.max_stable_dt()

        st.metric('网格尺寸', f'{nx} × {ny}')

        use_custom_dt = st.checkbox('自定义时间步长', value=False, key='use_custom_dt')
        if use_custom_dt:
            dt_val = st.number_input('时间步长 (s)', value=max_dt * 0.9,
                                     min_value=1e-18, max_value=1e-6,
                                     format='%.3e', key='dt_input')
            config.dt = dt_val
        else:
            config.dt = None

        courant = config.courant_number() if config.dt is not None else 0.7071
        stable = config.is_stable() if config.dt is not None else True

        if config.dt is not None:
            if stable:
                st.success(f'✅ Courant数: {courant:.4f} (稳定)')
            else:
                st.error(f'❌ Courant数: {courant:.4f} (不稳定! 最大dt={max_dt:.3e}s)')
        else:
            info_dt = max_dt * 0.9
            info_courant = C0 * info_dt * np.sqrt(1 / config.dx ** 2 + 1 / config.dy ** 2)
            st.info(f'自动时间步长: {info_dt:.3e}s (Courant ≈ {info_courant:.3f})')

    with st.expander('🧱 材料库', expanded=False):
        mat_lib = st.session_state.material_lib
        mat_names = mat_lib.list_materials()
        selected_mat = st.selectbox('预设材料', mat_names, key='mat_select')

        mat = mat_lib.get_material(selected_mat)
        st.write(f'相对介电常数: {mat.epsilon_r}')
        st.write(f'电导率: {mat.sigma} S/m')

        st.divider()
        st.subheader('自定义材料')
        new_mat_name = st.text_input('材料名称', value='MyMaterial', key='new_mat_name')
        new_eps_r = st.number_input('相对介电常数', value=2.0, min_value=1.0, max_value=1000.0, key='new_eps_r')
        new_sigma = st.number_input('电导率 (S/m)', value=0.0, min_value=0.0, max_value=1e10,
                                    format='%.1e', key='new_sigma')
        new_color = st.color_picker('颜色', '#888888', key='new_color')

        if st.button('添加材料', key='add_mat_btn'):
            new_mat = Material(new_mat_name, new_eps_r, new_sigma, color=new_color)
            mat_lib.add_material(new_mat)
            st.success(f'已添加材料: {new_mat_name}')
            st.rerun()

    with st.expander('🎨 结构绘制', expanded=True):
        struct_mgr = st.session_state.structure_mgr
        mat_names = mat_lib.list_materials()

        shape_type = st.selectbox('形状类型', ['rectangle', 'circle', 'line'],
                                  format_func=lambda x: {'rectangle': '矩形', 'circle': '圆形', 'line': '线'}[x],
                                  key='shape_type')

        struct_mat = st.selectbox('材料', mat_names, key='struct_mat')
        is_pec = st.checkbox('PEC (理想电导体)', value=False, key='struct_is_pec')

        params = {}
        if shape_type == 'rectangle':
            col1, col2 = st.columns(2)
            with col1:
                params['x0'] = st.number_input('X0', value=20.0, key='rect_x0') / factor
                params['y0'] = st.number_input('Y0', value=20.0, key='rect_y0') / factor
            with col2:
                params['width'] = st.number_input('宽度', value=20.0, key='rect_w') / factor
                params['height'] = st.number_input('高度', value=20.0, key='rect_h') / factor
        elif shape_type == 'circle':
            col1, col2 = st.columns(2)
            with col1:
                params['cx'] = st.number_input('中心X', value=50.0, key='circ_cx') / factor
                params['cy'] = st.number_input('中心Y', value=50.0, key='circ_cy') / factor
            with col2:
                params['radius'] = st.number_input('半径', value=15.0, key='circ_r') / factor
        elif shape_type == 'line':
            col1, col2 = st.columns(2)
            with col1:
                params['x0'] = st.number_input('X0', value=10.0, key='line_x0') / factor
                params['y0'] = st.number_input('Y0', value=10.0, key='line_y0') / factor
            with col2:
                params['x1'] = st.number_input('X1', value=90.0, key='line_x1') / factor
                params['y1'] = st.number_input('Y1', value=90.0, key='line_y1') / factor
            params['thickness'] = st.number_input('厚度', value=1.0, key='line_thick') / factor

        if st.button('➕ 添加结构', key='add_struct_btn', use_container_width=True):
            struct = Structure(shape_type, struct_mat, params, is_pec=is_pec)
            struct_mgr.add_structure(struct)
            st.success(f'已添加 {shape_type} 结构')
            st.rerun()

        st.divider()
        st.subheader('结构列表')
        if struct_mgr.structures:
            for idx, s in enumerate(struct_mgr.structures):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f'{idx+1}. {s.shape_type} - {s.material_name}')
                with col2:
                    if st.button('删除', key=f'del_struct_{idx}'):
                        struct_mgr.remove_structure(idx)
                        st.rerun()
        else:
            st.info('暂无结构')

        if st.button('🗑️ 清空所有结构', key='clear_structs'):
            struct_mgr.clear()
            st.rerun()

    with st.expander('⚡ 激励源', expanded=True):
        source_mgr = st.session_state.source_mgr
        source_types = ['point', 'line']
        source_type = st.selectbox('源类型', source_types,
                                   format_func=lambda x: {'point': '点源', 'line': '线源'}[x],
                                   key='source_type')

        waveform_type = st.selectbox('波形', ['gaussian', 'sine'],
                                     format_func=lambda x: {'gaussian': '高斯脉冲', 'sine': '正弦波'}[x],
                                     key='waveform_type')

        col1, col2 = st.columns(2)
        with col1:
            src_freq = st.number_input('频率 (THz)', value=10.0, min_value=0.1, max_value=1000.0, key='src_freq')
            src_amp = st.number_input('幅值', value=10.0, min_value=0.01, max_value=1000.0, key='src_amp')
        with col2:
            if waveform_type == 'gaussian':
                src_bw = st.number_input('带宽 (THz)', value=5.0, min_value=0.1, max_value=500.0, key='src_bw')

        col1, col2 = st.columns(2)
        with col1:
            src_x = st.number_input('位置 X', value=int(nx / 2), min_value=0, max_value=nx - 1, key='src_x')
        with col2:
            src_y = st.number_input('位置 Y', value=int(ny / 2), min_value=0, max_value=ny - 1, key='src_y')

        src_direction = 'x'
        if source_type == 'line':
            src_direction = st.selectbox('方向', ['x', 'y'], key='src_dir')

        if st.button('➕ 添加源', key='add_src_btn', use_container_width=True):
            wf = Waveform(
                waveform_type=waveform_type,
                frequency=src_freq * 1e12,
                amplitude=src_amp,
                bandwidth=src_bw * 1e12 if waveform_type == 'gaussian' else 0.0
            )
            src = Source(
                source_type=source_type,
                position=(int(src_x), int(src_y)),
                waveform=wf,
                direction=src_direction
            )
            source_mgr.add_source(src)
            st.success('已添加源')
            st.rerun()

        st.divider()
        st.subheader('平面波 (TFSF)')
        use_tfsf = st.checkbox('启用平面波', value=False, key='use_tfsf')

        if use_tfsf:
            tfsf_angle = st.selectbox('入射角度', [0, 90, 180, 270],
                                      format_func=lambda x: f'{x}°',
                                      key='tfsf_angle')
            tfsf_freq = st.number_input('频率 (THz)', value=10.0, key='tfsf_freq')
            tfsf_bw = st.number_input('带宽 (THz)', value=5.0, key='tfsf_bw')
            tfsf_amp = st.number_input('幅值', value=10.0, key='tfsf_amp')

            tfsf_wf = Waveform(
                waveform_type='gaussian',
                frequency=tfsf_freq * 1e12,
                amplitude=tfsf_amp,
                bandwidth=tfsf_bw * 1e12
            )
            tfsf = TFSF(
                x_min=5, x_max=nx - 6,
                y_min=5, y_max=ny - 6,
                incident_angle=tfsf_angle,
                waveform=tfsf_wf
            )
            source_mgr.set_tfsf(tfsf)
        else:
            source_mgr.set_tfsf(None)

        st.divider()
        st.subheader('源列表')
        if source_mgr.sources:
            for idx, s in enumerate(source_mgr.sources):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f'{idx+1}. {s.source_type} - {s.waveform.waveform_type}')
                with col2:
                    if st.button('删除', key=f'del_src_{idx}'):
                        source_mgr.remove_source(idx)
                        st.rerun()
        else:
            st.info('暂无点源/线源')

    with st.expander('🔲 边界条件', expanded=False):
        boundary = st.session_state.boundary
        bc_types = ['pec', 'mur', 'pml']
        bc_labels = {'pec': 'PEC反射壁', 'mur': 'Mur吸收', 'pml': 'PML完美匹配层'}

        col1, col2 = st.columns(2)
        with col1:
            boundary.x_min_type = st.selectbox('左边界', bc_types,
                                                index=bc_types.index(boundary.x_min_type),
                                                format_func=lambda x: bc_labels[x],
                                                key='bc_xmin')
            boundary.x_max_type = st.selectbox('右边界', bc_types,
                                                index=bc_types.index(boundary.x_max_type),
                                                format_func=lambda x: bc_labels[x],
                                                key='bc_xmax')
        with col2:
            boundary.y_min_type = st.selectbox('下边界', bc_types,
                                                index=bc_types.index(boundary.y_min_type),
                                                format_func=lambda x: bc_labels[x],
                                                key='bc_ymin')
            boundary.y_max_type = st.selectbox('上边界', bc_types,
                                                index=bc_types.index(boundary.y_max_type),
                                                format_func=lambda x: bc_labels[x],
                                                key='bc_ymax')

        boundary.pml_layers = st.number_input('PML层数', value=boundary.pml_layers,
                                              min_value=2, max_value=20, key='pml_layers')

    if compare_mode:
        with st.expander('🔄 对比仿真配置', expanded=True):
            st.caption('配置右侧对比面板的参数')

            st.subheader('边界条件')
            compare_bc = st.session_state.compare_boundary
            bc_types = ['pec', 'mur', 'pml']
            bc_labels = {'pec': 'PEC反射壁', 'mur': 'Mur吸收', 'pml': 'PML完美匹配层'}

            col1, col2 = st.columns(2)
            with col1:
                compare_bc.x_min_type = st.selectbox('左边界(对比)', bc_types,
                                                     index=bc_types.index(compare_bc.x_min_type),
                                                     format_func=lambda x: bc_labels[x],
                                                     key='compare_bc_xmin')
                compare_bc.x_max_type = st.selectbox('右边界(对比)', bc_types,
                                                     index=bc_types.index(compare_bc.x_max_type),
                                                     format_func=lambda x: bc_labels[x],
                                                     key='compare_bc_xmax')
            with col2:
                compare_bc.y_min_type = st.selectbox('下边界(对比)', bc_types,
                                                     index=bc_types.index(compare_bc.y_min_type),
                                                     format_func=lambda x: bc_labels[x],
                                                     key='compare_bc_ymin')
                compare_bc.y_max_type = st.selectbox('上边界(对比)', bc_types,
                                                     index=bc_types.index(compare_bc.y_max_type),
                                                     format_func=lambda x: bc_labels[x],
                                                     key='compare_bc_ymax')

            compare_bc.pml_layers = st.number_input('PML层数(对比)', value=compare_bc.pml_layers,
                                                    min_value=2, max_value=20, key='compare_pml_layers')

            st.divider()
            st.subheader('结构与材料')

            copy_structures = st.button('📋 复制基准结构到对比', key='copy_struct_to_compare')
            if copy_structures:
                st.session_state.compare_structure_mgr = copy.deepcopy(st.session_state.structure_mgr)
                st.session_state.compare_material_lib = copy.deepcopy(st.session_state.material_lib)
                st.success('已复制基准结构和材料')

            compare_struct_mgr = st.session_state.compare_structure_mgr
            compare_mat_lib = st.session_state.compare_material_lib
            compare_mat_names = compare_mat_lib.list_materials()

            compare_shape_type = st.selectbox('形状类型(对比)', ['rectangle', 'circle', 'line'],
                                              format_func=lambda x: {'rectangle': '矩形', 'circle': '圆形', 'line': '线'}[x],
                                              key='compare_shape_type')

            compare_struct_mat = st.selectbox('材料(对比)', compare_mat_names, key='compare_struct_mat')
            compare_is_pec = st.checkbox('PEC (理想电导体)', value=False, key='compare_struct_is_pec')

            compare_params = {}
            if compare_shape_type == 'rectangle':
                col1, col2 = st.columns(2)
                with col1:
                    compare_params['x0'] = st.number_input('X0', value=20.0, key='compare_rect_x0') / factor
                    compare_params['y0'] = st.number_input('Y0', value=20.0, key='compare_rect_y0') / factor
                with col2:
                    compare_params['width'] = st.number_input('宽度', value=20.0, key='compare_rect_w') / factor
                    compare_params['height'] = st.number_input('高度', value=20.0, key='compare_rect_h') / factor
            elif compare_shape_type == 'circle':
                col1, col2 = st.columns(2)
                with col1:
                    compare_params['cx'] = st.number_input('中心X', value=50.0, key='compare_circ_cx') / factor
                    compare_params['cy'] = st.number_input('中心Y', value=50.0, key='compare_circ_cy') / factor
                with col2:
                    compare_params['radius'] = st.number_input('半径', value=15.0, key='compare_circ_r') / factor
            elif compare_shape_type == 'line':
                col1, col2 = st.columns(2)
                with col1:
                    compare_params['x0'] = st.number_input('X0', value=10.0, key='compare_line_x0') / factor
                    compare_params['y0'] = st.number_input('Y0', value=10.0, key='compare_line_y0') / factor
                with col2:
                    compare_params['x1'] = st.number_input('X1', value=90.0, key='compare_line_x1') / factor
                    compare_params['y1'] = st.number_input('Y1', value=90.0, key='compare_line_y1') / factor
                compare_params['thickness'] = st.number_input('厚度', value=1.0, key='compare_line_thick') / factor

            if st.button('➕ 添加对比结构', key='add_compare_struct_btn', use_container_width=True):
                compare_struct = Structure(compare_shape_type, compare_struct_mat, compare_params, is_pec=compare_is_pec)
                compare_struct_mgr.add_structure(compare_struct)
                st.success(f'已添加对比 {compare_shape_type} 结构')
                st.rerun()

            st.subheader('对比结构列表')
            if compare_struct_mgr.structures:
                for idx, s in enumerate(compare_struct_mgr.structures):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f'{idx+1}. {s.shape_type} - {s.material_name}')
                    with col2:
                        if st.button('删除', key=f'del_compare_struct_{idx}'):
                            compare_struct_mgr.remove_structure(idx)
                            st.rerun()
            else:
                st.info('暂无对比结构')

            if st.button('🗑️ 清空对比结构', key='clear_compare_structs'):
                compare_struct_mgr.clear()
                st.rerun()

            st.divider()
            st.subheader('激励源')

            copy_sources = st.button('📋 复制基准源到对比', key='copy_src_to_compare')
            if copy_sources:
                st.session_state.compare_source_mgr = copy.deepcopy(st.session_state.source_mgr)
                st.success('已复制基准源配置')

            compare_source_mgr = st.session_state.compare_source_mgr
            compare_source_types = ['point', 'line']
            compare_source_type = st.selectbox('源类型(对比)', compare_source_types,
                                               format_func=lambda x: {'point': '点源', 'line': '线源'}[x],
                                               key='compare_source_type')

            compare_waveform_type = st.selectbox('波形(对比)', ['gaussian', 'sine'],
                                                 format_func=lambda x: {'gaussian': '高斯脉冲', 'sine': '正弦波'}[x],
                                                 key='compare_waveform_type')

            col1, col2 = st.columns(2)
            with col1:
                compare_src_freq = st.number_input('频率 (THz)', value=10.0, min_value=0.1, max_value=1000.0, key='compare_src_freq')
                compare_src_amp = st.number_input('幅值', value=10.0, min_value=0.01, max_value=1000.0, key='compare_src_amp')
            with col2:
                if compare_waveform_type == 'gaussian':
                    compare_src_bw = st.number_input('带宽 (THz)', value=5.0, min_value=0.1, max_value=500.0, key='compare_src_bw')

            col1, col2 = st.columns(2)
            with col1:
                compare_src_x = st.number_input('位置 X', value=int(nx / 2), min_value=0, max_value=nx - 1, key='compare_src_x')
            with col2:
                compare_src_y = st.number_input('位置 Y', value=int(ny / 2), min_value=0, max_value=ny - 1, key='compare_src_y')

            compare_src_direction = 'x'
            if compare_source_type == 'line':
                compare_src_direction = st.selectbox('方向', ['x', 'y'], key='compare_src_dir')

            if st.button('➕ 添加对比源', key='add_compare_src_btn', use_container_width=True):
                compare_wf = Waveform(
                    waveform_type=compare_waveform_type,
                    frequency=compare_src_freq * 1e12,
                    amplitude=compare_src_amp,
                    bandwidth=compare_src_bw * 1e12 if compare_waveform_type == 'gaussian' else 0.0
                )
                compare_src = Source(
                    source_type=compare_source_type,
                    position=(int(compare_src_x), int(compare_src_y)),
                    waveform=compare_wf,
                    direction=compare_src_direction
                )
                compare_source_mgr.add_source(compare_src)
                st.success('已添加对比源')
                st.rerun()

            st.subheader('对比源列表')
            if compare_source_mgr.sources:
                for idx, s in enumerate(compare_source_mgr.sources):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f'{idx+1}. {s.source_type} - {s.waveform.waveform_type}')
                    with col2:
                        if st.button('删除', key=f'del_compare_src_{idx}'):
                            compare_source_mgr.remove_source(idx)
                            st.rerun()
            else:
                st.info('暂无对比源')

    with st.expander('📍 观测点 & 近场采集', expanded=False):
        obs_points = st.session_state.observation_points

        col1, col2 = st.columns(2)
        with col1:
            obs_x = st.number_input('观测点 X', value=int(nx / 2), min_value=0, max_value=nx - 1, key='obs_x')
        with col2:
            obs_y = st.number_input('观测点 Y', value=int(ny / 2), min_value=0, max_value=ny - 1, key='obs_y')

        if st.button('➕ 添加观测点', key='add_obs'):
            pt = (int(obs_x), int(obs_y))
            if pt not in obs_points:
                obs_points.append(pt)
                st.session_state.config.observation_points = obs_points
                st.success(f'已添加观测点 {pt}')

        if obs_points:
            st.write('观测点列表:')
            for idx, pt in enumerate(obs_points):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f'{idx+1}. ({pt[0]}, {pt[1]})')
                with col2:
                    if st.button('删除', key=f'del_obs_{idx}'):
                        obs_points.pop(idx)
                        st.session_state.config.observation_points = obs_points
                        st.rerun()

        st.divider()
        st.subheader('近场采集面')
        use_nf = st.checkbox('启用近远场变换', value=False, key='use_nf')

        if use_nf:
            col1, col2 = st.columns(2)
            with col1:
                nf_x1 = st.number_input('X1', value=10, min_value=0, max_value=nx - 1, key='nf_x1')
                nf_y1 = st.number_input('Y1', value=10, min_value=0, max_value=ny - 1, key='nf_y1')
            with col2:
                nf_x2 = st.number_input('X2', value=nx - 11, min_value=0, max_value=nx - 1, key='nf_x2')
                nf_y2 = st.number_input('Y2', value=ny - 11, min_value=0, max_value=ny - 1, key='nf_y2')
            st.session_state.near_field_box = (int(nf_x1), int(nf_y1), int(nf_x2), int(nf_y2))
            st.session_state.config.near_field_box = (int(nf_x1), int(nf_y1), int(nf_x2), int(nf_y2))
        else:
            st.session_state.near_field_box = None
            st.session_state.config.near_field_box = None

    with st.expander('📋 模板', expanded=False):
        templates = list_templates()
        template_names = [t['name'] for t in templates]
        template_keys = [t['key'] for t in templates]

        selected_template = st.selectbox('选择模板', template_names, key='template_select')

        if st.button('🎯 应用模板', key='apply_template', use_container_width=True):
            idx = template_names.index(selected_template)
            tpl = get_template(template_keys[idx])
            st.session_state.config = tpl.config
            st.session_state.material_lib = tpl.material_lib
            st.session_state.structure_mgr = tpl.structure_mgr
            st.session_state.source_mgr = tpl.source_mgr
            st.session_state.boundary = tpl.boundary
            st.session_state.observation_points = tpl.observation_points
            st.session_state.config.observation_points = tpl.observation_points
            st.session_state.near_field_box = tpl.near_field_box
            st.session_state.config.near_field_box = tpl.near_field_box

            factor = factor_to_unit(tpl.config.unit)

            st.session_state.unit_select = tpl.config.unit
            st.session_state.domain_width = tpl.config.width * factor
            st.session_state.domain_height = tpl.config.height * factor
            st.session_state.dx_input = tpl.config.dx * factor
            st.session_state.total_steps = tpl.config.total_time_steps
            st.session_state.sample_interval = tpl.config.sample_interval
            st.session_state.use_custom_dt = False

            st.session_state.bc_xmin = tpl.boundary.x_min_type
            st.session_state.bc_xmax = tpl.boundary.x_max_type
            st.session_state.bc_ymin = tpl.boundary.y_min_type
            st.session_state.bc_ymax = tpl.boundary.y_max_type
            st.session_state.pml_layers = tpl.boundary.pml_layers

            st.session_state.use_tfsf = tpl.source_mgr.tfsf is not None
            if tpl.source_mgr.tfsf is not None:
                st.session_state.tfsf_angle = tpl.source_mgr.tfsf.incident_angle
                st.session_state.tfsf_freq = tpl.source_mgr.tfsf.waveform.frequency / 1e12
                st.session_state.tfsf_bw = tpl.source_mgr.tfsf.waveform.bandwidth / 1e12
                st.session_state.tfsf_amp = tpl.source_mgr.tfsf.waveform.amplitude

            st.session_state.result = None
            st.session_state.far_field_result = None
            st.session_state.current_frame = 0

            st.success(f'已应用模板: {selected_template}')
            st.rerun()

    with st.expander('📤 导入/导出', expanded=False):
        config_json = export_config_json(
            st.session_state.config,
            st.session_state.material_lib,
            st.session_state.structure_mgr,
            st.session_state.source_mgr,
            st.session_state.boundary
        )
        st.download_button(
            label='💾 导出配置 JSON',
            data=config_json,
            file_name='fdtd_config.json',
            mime='application/json',
            use_container_width=True
        )

        uploaded_file = st.file_uploader('📂 导入配置 JSON', type=['json'], key='config_upload')
        if uploaded_file is not None:
            try:
                json_str = uploaded_file.getvalue().decode('utf-8')
                cfg, ml, sm, srcm, bc = import_config_json(json_str)
                st.session_state.config = cfg
                st.session_state.material_lib = ml
                st.session_state.structure_mgr = sm
                st.session_state.source_mgr = srcm
                st.session_state.boundary = bc
                st.success('配置导入成功!')
                st.rerun()
            except Exception as e:
                st.error(f'导入失败: {e}')

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        run_disabled = (st.session_state.config.dt is not None and
                        not st.session_state.config.is_stable())
        run_label = '▶️ 运行对比仿真' if compare_mode else '▶️ 运行仿真'
        if st.button(run_label, key='run_btn', use_container_width=True,
                     type='primary', disabled=run_disabled):
            with st.spinner('正在运行仿真...'):
                progress_bar = st.progress(0)

                def progress_cb(current, total):
                    progress_bar.progress(current / total)

                config_copy = copy.deepcopy(st.session_state.config)
                config_copy.observation_points = st.session_state.observation_points
                config_copy.near_field_box = st.session_state.near_field_box
                mat_lib_copy = copy.deepcopy(st.session_state.material_lib)
                struct_copy = copy.deepcopy(st.session_state.structure_mgr)
                src_copy = copy.deepcopy(st.session_state.source_mgr)
                bc_copy = copy.deepcopy(st.session_state.boundary)

                fdtd = FDTD2D(config_copy, mat_lib_copy, struct_copy, src_copy, bc_copy)
                result = fdtd.run(progress_callback=progress_cb)
                st.session_state.result = result
                st.session_state.current_frame = 0
                st.session_state.color_grid = fdtd.get_color_grid_physical()

                if result.near_field_data is not None and st.session_state.near_field_box is not None:
                    try:
                        nf_transform = NearFarFieldTransform(
                            result.near_field_data,
                            config_copy.dx, config_copy.dy,
                            config_copy.dt if config_copy.dt else config_copy.max_stable_dt() * 0.9,
                            st.session_state.near_field_box,
                            10e9
                        )
                        angles, ff_mag, rcs_db = nf_transform.compute_far_field()
                        st.session_state.far_field_result = (angles, ff_mag, rcs_db)
                    except Exception as e:
                        st.warning(f'近远场变换失败: {e}')
                        st.session_state.far_field_result = None
                else:
                    st.session_state.far_field_result = None

                if compare_mode:
                    progress_bar.progress(0.5)
                    compare_config_copy = copy.deepcopy(st.session_state.config)
                    compare_config_copy.observation_points = st.session_state.observation_points
                    compare_config_copy.near_field_box = st.session_state.near_field_box
                    compare_mat_lib_copy = copy.deepcopy(st.session_state.compare_material_lib)
                    compare_struct_copy = copy.deepcopy(st.session_state.compare_structure_mgr)
                    compare_src_copy = copy.deepcopy(st.session_state.compare_source_mgr)
                    compare_bc_copy = copy.deepcopy(st.session_state.compare_boundary)

                    compare_fdtd = FDTD2D(compare_config_copy, compare_mat_lib_copy,
                                          compare_struct_copy, compare_src_copy, compare_bc_copy)
                    compare_result = compare_fdtd.run(progress_callback=progress_cb)
                    st.session_state.compare_result = compare_result
                    st.session_state.compare_color_grid = compare_fdtd.get_color_grid_physical()

                progress_bar.empty()
                if compare_mode:
                    st.success(f'对比仿真完成! 基准: {result.computation_time:.2f}s, 对比: {compare_result.computation_time:.2f}s')
                else:
                    st.success(f'仿真完成! 耗时: {result.computation_time:.2f}s')
                st.rerun()

    with col2:
        if st.button('🔄 重置', key='reset_btn', use_container_width=True):
            st.session_state.result = None
            st.session_state.far_field_result = None
            st.session_state.compare_result = None
            st.session_state.current_frame = 0
            st.session_state.bookmarks = []
            st.session_state.convergence_result = None
            st.rerun()

st.title('📡 FDTD 电磁场仿真与分析工具')
st.caption('基于 Yee 网格的二维 TM 模式有限差分时域仿真')

result = st.session_state.result
config = st.session_state.config
factor = factor_to_unit(config.unit)

if result is None:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader('结构预览')
        try:
            fdtd_preview = FDTD2D(
                copy.deepcopy(config),
                copy.deepcopy(st.session_state.material_lib),
                copy.deepcopy(st.session_state.structure_mgr),
                copy.deepcopy(st.session_state.source_mgr),
                copy.deepcopy(st.session_state.boundary)
            )
            color_grid = fdtd_preview.get_color_grid_physical()
            fig, ax = plt.subplots(figsize=(8, 6))
            extent = [0, config.ny * config.dy * factor, 0, config.nx * config.dx * factor]
            ax.imshow(color_grid.transpose(1, 0, 2), extent=extent, origin='lower')
            ax.set_xlabel(f'Y ({config.unit})')
            ax.set_ylabel(f'X ({config.unit})')
            ax.set_title('材料结构分布')
            ax.set_aspect('equal')

            obs_pts = st.session_state.observation_points
            if obs_pts:
                for pt in obs_pts:
                    ax.plot(pt[1] * config.dy * factor, pt[0] * config.dx * factor,
                            'ro', markersize=8, markeredgecolor='white')

            if st.session_state.near_field_box is not None:
                x1, y1, x2, y2 = st.session_state.near_field_box
                rect = plt.Rectangle((y1 * config.dy * factor, x1 * config.dx * factor),
                                     (y2 - y1) * config.dy * factor,
                                     (x2 - x1) * config.dx * factor,
                                     fill=False, edgecolor='yellow', linestyle='--', linewidth=2)
                ax.add_patch(rect)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f'预览生成失败: {e}')

    with col_right:
        st.subheader('仿真信息')
        st.metric('网格尺寸', f'{config.nx} × {config.ny}')
        st.metric('总时间步数', config.total_time_steps)
        st.metric('结构数量', len(st.session_state.structure_mgr.structures))
        st.metric('源数量', len(st.session_state.source_mgr.sources))
        st.metric('观测点数', len(st.session_state.observation_points))

        st.divider()
        st.subheader('使用说明')
        st.markdown("""
        1. 在左侧配置仿真参数
        2. 添加材料结构
        3. 设置激励源
        4. 选择边界条件
        5. 点击"运行仿真"
        6. 查看结果和分析
        """)

else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['🖼️ 场分布', '📈 时域/频域', '📡 远场', '⚡ 参数扫描', '📐 收敛性分析'])

    with tab1:
        if compare_mode and st.session_state.compare_result is not None:
            compare_result = st.session_state.compare_result
            compare_color_grid = getattr(st.session_state, 'compare_color_grid', None)
            total_frames = min(len(result.ez_frames), len(compare_result.ez_frames))
            st.session_state.current_frame = min(
                st.session_state.get('current_frame', 0),
                total_frames - 1
            )
            current_frame = st.session_state.current_frame

            vmax_base = np.max(np.abs(result.ez_frames)) if np.max(np.abs(result.ez_frames)) > 0 else 1.0
            vmax_comp = np.max(np.abs(compare_result.ez_frames)) if np.max(np.abs(compare_result.ez_frames)) > 0 else 1.0
            vmax = max(vmax_base, vmax_comp)

            col_base, col_comp = st.columns(2)

            with col_base:
                st.subheader('🔵 基准仿真')
                color_grid = getattr(st.session_state, 'color_grid', None)
                fig_base = plot_field_heatmap(
                    result.ez_frames[current_frame],
                    title=f'基准 - 时间步 {current_frame * config.sample_interval}',
                    dx=config.dx, dy=config.dy, unit=config.unit,
                    color_grid=color_grid,
                    vmin=-vmax, vmax=vmax
                )
                st.pyplot(fig_base)
                plt.close(fig_base)

            with col_comp:
                st.subheader('🔴 对比仿真')
                fig_comp = plot_field_heatmap(
                    compare_result.ez_frames[current_frame],
                    title=f'对比 - 时间步 {current_frame * config.sample_interval}',
                    dx=config.dx, dy=config.dy, unit=config.unit,
                    color_grid=compare_color_grid,
                    vmin=-vmax, vmax=vmax
                )
                st.pyplot(fig_comp)
                plt.close(fig_comp)

            st.divider()

            col_play1, col_play2, col_play3, col_play4, col_play5 = st.columns([1, 1, 3, 1, 1.5])
            with col_play1:
                if st.button('⏮️', key='prev_frame'):
                    st.session_state.current_frame = max(0, st.session_state.current_frame - 1)
                    st.rerun()
            with col_play2:
                if st.button('▶️' if not st.session_state.is_playing else '⏸️',
                             key='play_pause'):
                    st.session_state.is_playing = not st.session_state.is_playing
                    st.rerun()
            with col_play3:
                frame_idx = st.slider('帧 (联动)', min_value=0, max_value=total_frames - 1,
                                      key='current_frame')
            with col_play4:
                if st.button('⏭️', key='next_frame'):
                    st.session_state.current_frame = min(total_frames - 1, st.session_state.current_frame + 1)
                    st.rerun()
            with col_play5:
                if st.button('📸 保存快照', key='save_bookmark_btn', use_container_width=True):
                    if len(st.session_state.bookmarks) >= 10:
                        st.session_state.bookmarks.pop(0)
                    thumb_fig = plot_field_heatmap(
                        result.ez_frames[current_frame],
                        title=f'帧 {current_frame}',
                        dx=config.dx, dy=config.dy, unit=config.unit,
                        color_grid=color_grid,
                        vmin=-vmax, vmax=vmax
                    )
                    thumb_buf = io.BytesIO()
                    thumb_fig.savefig(thumb_buf, format='png', dpi=50, bbox_inches='tight')
                    thumb_buf.seek(0)
                    st.session_state.bookmarks.append({
                        'frame': current_frame,
                        'thumbnail': thumb_buf.getvalue(),
                        'note': f'帧 {current_frame}'
                    })
                    plt.close(thumb_fig)
                    st.success('已保存快照')
                    st.rerun()

            if st.session_state.is_playing:
                import time as tm
                next_frame = (st.session_state.current_frame + 1) % total_frames
                st.session_state.current_frame = next_frame
                tm.sleep(0.08)
                st.rerun()

            st.divider()
            st.subheader('🔖 书签列表')
            if st.session_state.bookmarks:
                bookmark_cols = st.columns(5)
                for i, bm in enumerate(st.session_state.bookmarks):
                    with bookmark_cols[i % 5]:
                        st.image(bm['thumbnail'], caption=f'帧 {bm["frame"]}', use_column_width=True)
                        new_note = st.text_input('备注', value=bm['note'], key=f'note_{i}',
                                                 label_visibility='collapsed')
                        bm['note'] = new_note
                        col_btn1, col_btn2 = st.columns(2)
                        with col_btn1:
                            if st.button('跳转', key=f'jump_{i}', use_container_width=True):
                                st.session_state.current_frame = bm['frame']
                                st.rerun()
                        with col_btn2:
                            if st.button('删除', key=f'del_bm_{i}', use_container_width=True):
                                st.session_state.bookmarks.pop(i)
                                st.rerun()
            else:
                st.info('暂无书签，点击"保存快照"添加')

            st.divider()
            st.subheader('📊 时域波形 & 频谱对比')
            obs_points = st.session_state.observation_points
            if obs_points and result.observation_data and compare_result.observation_data:
                selected_idx = st.selectbox('选择观测点',
                                            [f'点 {i+1}: ({pt[0]}, {pt[1]})' for i, pt in enumerate(obs_points)],
                                            key='compare_obs_select')
                idx = [f'点 {i+1}: ({pt[0]}, {pt[1]})' for i, pt in enumerate(obs_points)].index(selected_idx)
                pt = obs_points[idx]
                pt_key = tuple(pt)

                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    if pt_key in result.observation_data:
                        ax.plot(result.observation_times * 1e9, result.observation_data[pt_key],
                                'b-', linewidth=1, label='基准')
                    if pt_key in compare_result.observation_data:
                        ax.plot(compare_result.observation_times * 1e9, compare_result.observation_data[pt_key],
                                'r--', linewidth=1, label='对比')
                    ax.set_xlabel('Time (ns)')
                    ax.set_ylabel('Ez (V/m)')
                    ax.set_title(f'时域波形对比 - 观测点 {pt}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                with col2:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    if pt_key in result.observation_data:
                        data_base = result.observation_data[pt_key]
                        n_base = len(data_base)
                        dt_base = result.observation_times[1] - result.observation_times[0]
                        freq_base = np.fft.fftfreq(n_base, dt_base)
                        spec_base = np.abs(np.fft.fft(data_base))
                        spec_db_base = 20 * np.log10(spec_base + 1e-30)
                        pos_mask_base = freq_base >= 0
                        ax.plot(freq_base[pos_mask_base] / 1e9, spec_db_base[pos_mask_base],
                                'b-', linewidth=1, label='基准')

                    if pt_key in compare_result.observation_data:
                        data_comp = compare_result.observation_data[pt_key]
                        n_comp = len(data_comp)
                        dt_comp = compare_result.observation_times[1] - compare_result.observation_times[0]
                        freq_comp = np.fft.fftfreq(n_comp, dt_comp)
                        spec_comp = np.abs(np.fft.fft(data_comp))
                        spec_db_comp = 20 * np.log10(spec_comp + 1e-30)
                        pos_mask_comp = freq_comp >= 0
                        ax.plot(freq_comp[pos_mask_comp] / 1e9, spec_db_comp[pos_mask_comp],
                                'r--', linewidth=1, label='对比')

                    ax.set_xlabel('Frequency (GHz)')
                    ax.set_ylabel('Magnitude (dB)')
                    ax.set_title(f'频谱对比 - 观测点 {pt}')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.info('请先添加观测点')

        else:
            col_left, col_right = st.columns([3, 1])

            with col_left:
                st.subheader('Ez 场分布')

                total_frames = len(result.ez_frames)
                st.session_state.current_frame = min(
                    st.session_state.get('current_frame', 0),
                    total_frames - 1
                )

                current_frame = st.session_state.current_frame
                ez_frame = result.ez_frames[current_frame]
                vmax = np.max(np.abs(result.ez_frames)) if np.max(np.abs(result.ez_frames)) > 0 else 1.0

                color_grid = getattr(st.session_state, 'color_grid', None)

                plot_placeholder = st.empty()
                fig = plot_field_heatmap(
                    ez_frame,
                    title=f'Ez Field - 时间步 {current_frame * config.sample_interval}',
                    dx=config.dx, dy=config.dy, unit=config.unit,
                    color_grid=color_grid,
                    vmin=-vmax, vmax=vmax
                )
                plot_placeholder.pyplot(fig)
                plt.close(fig)

                st.divider()

                col_play1, col_play2, col_play3, col_play4, col_play5 = st.columns([1, 1, 3, 1, 1.5])
                with col_play1:
                    if st.button('⏮️', key='prev_frame'):
                        st.session_state.current_frame = max(0, st.session_state.current_frame - 1)
                        st.rerun()
                with col_play2:
                    if st.button('▶️' if not st.session_state.is_playing else '⏸️',
                                 key='play_pause'):
                        st.session_state.is_playing = not st.session_state.is_playing
                        st.rerun()
                with col_play3:
                    frame_idx = st.slider('帧', min_value=0, max_value=total_frames - 1,
                                          key='current_frame')
                with col_play4:
                    if st.button('⏭️', key='next_frame'):
                        st.session_state.current_frame = min(total_frames - 1, st.session_state.current_frame + 1)
                        st.rerun()
                with col_play5:
                    if st.button('📸 保存快照', key='save_bookmark_btn', use_container_width=True):
                        if len(st.session_state.bookmarks) >= 10:
                            st.session_state.bookmarks.pop(0)
                        thumb_fig = plot_field_heatmap(
                            ez_frame,
                            title=f'帧 {current_frame}',
                            dx=config.dx, dy=config.dy, unit=config.unit,
                            color_grid=color_grid,
                            vmin=-vmax, vmax=vmax
                        )
                        thumb_buf = io.BytesIO()
                        thumb_fig.savefig(thumb_buf, format='png', dpi=50, bbox_inches='tight')
                        thumb_buf.seek(0)
                        st.session_state.bookmarks.append({
                            'frame': current_frame,
                            'thumbnail': thumb_buf.getvalue(),
                            'note': f'帧 {current_frame}'
                        })
                        plt.close(thumb_fig)
                        st.success('已保存快照')
                        st.rerun()

                if st.session_state.is_playing:
                    import time as tm
                    next_frame = (st.session_state.current_frame + 1) % total_frames
                    st.session_state.current_frame = next_frame
                    tm.sleep(0.08)
                    st.rerun()

                st.divider()
                st.subheader('🔖 书签列表')
                if st.session_state.bookmarks:
                    bookmark_cols = st.columns(5)
                    for i, bm in enumerate(st.session_state.bookmarks):
                        with bookmark_cols[i % 5]:
                            st.image(bm['thumbnail'], caption=f'帧 {bm["frame"]}', use_column_width=True)
                            new_note = st.text_input('备注', value=bm['note'], key=f'note_{i}',
                                                     label_visibility='collapsed')
                            bm['note'] = new_note
                            col_btn1, col_btn2 = st.columns(2)
                            with col_btn1:
                                if st.button('跳转', key=f'jump_{i}', use_container_width=True):
                                    st.session_state.current_frame = bm['frame']
                                    st.rerun()
                            with col_btn2:
                                if st.button('删除', key=f'del_bm_{i}', use_container_width=True):
                                    st.session_state.bookmarks.pop(i)
                                    st.rerun()
                else:
                    st.info('暂无书签，点击"保存快照"添加')

            with col_right:
                st.subheader('能量分布')
                fig_energy = plot_energy_density(
                    result.energy_density,
                    dx=config.dx, dy=config.dy, unit=config.unit,
                    title='时间平均能量密度'
                )
                st.pyplot(fig_energy)
                plt.close(fig_energy)

                st.divider()
                st.subheader('仿真统计')
                st.metric('计算耗时', f'{result.computation_time:.2f} s')
                st.metric('采样帧数', len(result.ez_frames))
                st.metric('峰值场强', f'{np.max(np.abs(result.ez_final)):.4f} V/m')

    with tab2:
        obs_points = st.session_state.observation_points
        if obs_points and result.observation_data:
            selected_idx = st.selectbox('选择观测点',
                                        [f'点 {i+1}: ({pt[0]}, {pt[1]})' for i, pt in enumerate(obs_points)],
                                        key='obs_select')
            idx = [f'点 {i+1}: ({pt[0]}, {pt[1]})' for i, pt in enumerate(obs_points)].index(selected_idx)
            pt = obs_points[idx]
            pt_key = tuple(pt)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader('时域波形')
                if pt_key in result.observation_data:
                    fig_time = plot_time_waveform(
                        result.observation_times,
                        result.observation_data[pt_key],
                        point_label=f'({pt[0]}, {pt[1]})',
                        title='Ez 随时间变化'
                    )
                    st.pyplot(fig_time)
                    plt.close(fig_time)
                else:
                    st.info('无观测数据')

            with col2:
                st.subheader('频谱分析')
                if pt_key in result.observation_data:
                    fig_spec = plot_frequency_spectrum(
                        result.observation_times,
                        result.observation_data[pt_key],
                        title='频率响应'
                    )
                    st.pyplot(fig_spec)
                    plt.close(fig_spec)
                else:
                    st.info('无观测数据')

            st.divider()
            st.subheader('所有观测点对比')
            fig, ax = plt.subplots(figsize=(10, 4))
            for i, opt in enumerate(obs_points):
                opt_key = tuple(opt)
                if opt_key in result.observation_data:
                    ax.plot(result.observation_times * 1e9, result.observation_data[opt_key],
                            label=f'点{i+1} ({opt[0]},{opt[1]})', linewidth=1)
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel('Ez (V/m)')
            ax.set_title('多观测点对比')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info('请先添加观测点')

    with tab3:
        if st.session_state.far_field_result is not None:
            angles, ff_mag, rcs_db = st.session_state.far_field_result

            col1, col2 = st.columns(2)

            with col1:
                st.subheader('远场方向图 (极坐标)')
                fig_polar = plot_far_field_polar(angles, ff_mag, title='Far Field Pattern')
                st.pyplot(fig_polar)
                plt.close(fig_polar)

            with col2:
                st.subheader('双站 RCS')
                fig_rcs = plot_rcs(angles, rcs_db, title='Bistatic RCS')
                st.pyplot(fig_rcs)
                plt.close(fig_rcs)

            st.divider()
            st.metric('最大 RCS', f'{np.max(rcs_db):.2f} dBsm')
            st.metric('最小 RCS', f'{np.min(rcs_db):.2f} dBsm')

            if st.button('📥 导出远场数据 CSV', key='export_ff_csv'):
                csv_data = export_far_field_csv(angles, ff_mag, rcs_db)
                st.download_button(
                    '下载 CSV',
                    data=csv_data,
                    file_name='far_field.csv',
                    mime='text/csv'
                )
        else:
            st.info('请在侧边栏启用近远场变换后重新运行仿真')

    with tab4:
        st.subheader('参数扫描')

        param_options = [
            ('config.dx', '空间步长 dx'),
            ('config.total_time_steps', '总时间步数'),
            ('boundary.pml_layers', 'PML层数'),
        ]

        if st.session_state.source_mgr.sources:
            for i, s in enumerate(st.session_state.source_mgr.sources):
                param_options.append((f'source.{i}.waveform.frequency', f'源{i+1}频率'))

        col1, col2 = st.columns(2)
        with col1:
            selected_param = st.selectbox(
                '扫描参数',
                [p[1] for p in param_options],
                key='sweep_param'
            )
            param_idx = [p[1] for p in param_options].index(selected_param)
            param_path = param_options[param_idx][0]

            start_val = st.number_input('起始值', value=1.0, key='sweep_start')
            end_val = st.number_input('结束值', value=5.0, key='sweep_end')
            num_steps = st.number_input('步数', value=5, min_value=2, max_value=20, key='sweep_steps')

        metric_options = [
            ('peak_ez', '观测点峰值场强'),
            ('peak_energy', '峰值能量密度'),
            ('total_energy', '总能量'),
        ]
        with col2:
            selected_metric = st.selectbox(
                '评估指标',
                [m[1] for m in metric_options],
                key='sweep_metric'
            )
            metric_idx = [m[1] for m in metric_options].index(selected_metric)
            metric_name = metric_options[metric_idx][0]

            metric_params = {}
            if obs_points and metric_name == 'peak_ez':
                metric_params['point'] = obs_points[0]

        if st.button('▶️ 运行参数扫描', key='run_sweep', type='primary'):
            with st.spinner('正在进行参数扫描...'):
                actual_start = start_val
                actual_end = end_val
                if 'frequency' in param_path:
                    actual_start = start_val * 1e12
                    actual_end = end_val * 1e12
                    st.info(f'频率参数已从 THz 转换为 Hz: {start_val}-{end_val} THz → {actual_start:.2e}-{actual_end:.2e} Hz')

                sweep_cfg = SweepConfig(
                    param_name=selected_param,
                    param_path=param_path,
                    start_value=actual_start,
                    end_value=actual_end,
                    num_steps=int(num_steps),
                    metric=metric_name,
                    metric_params=metric_params
                )

                try:
                    sweep = ParametricSweep(
                        copy.deepcopy(config),
                        copy.deepcopy(st.session_state.material_lib),
                        copy.deepcopy(st.session_state.structure_mgr),
                        copy.deepcopy(st.session_state.source_mgr),
                        copy.deepcopy(st.session_state.boundary)
                    )

                    progress_bar = st.progress(0)

                    def sweep_cb(current, total):
                        progress_bar.progress(current / total)

                    param_vals, metric_vals, results = sweep.run(sweep_cfg, progress_callback=sweep_cb)
                    progress_bar.empty()

                    st.session_state.sweep_result = (
                        param_vals, metric_vals, results,
                        selected_param, selected_metric
                    )
                    st.success('参数扫描完成!')
                except Exception as e:
                    st.error(f'参数扫描失败: {e}')

        if 'sweep_result' in st.session_state:
            if len(st.session_state.sweep_result) == 5:
                param_vals, metric_vals, results, disp_param, disp_metric = st.session_state.sweep_result
            else:
                param_vals, metric_vals, results = st.session_state.sweep_result
                disp_param, disp_metric = selected_param, selected_metric

            fig_sweep = plot_parametric_sweep(
                param_vals, metric_vals,
                param_name=disp_param,
                metric_name=disp_metric,
                title='参数扫描结果'
            )
            st.pyplot(fig_sweep)
            plt.close(fig_sweep)

    with tab5:
        st.subheader('网格收敛性分析')
        st.caption('通过不同网格密度的仿真结果对比，判断当前网格是否足够精细')

        col1, col2 = st.columns(2)
        with col1:
            ref_dx = st.number_input('参考网格步长 dx (um)', value=1.0, min_value=0.01,
                                     max_value=100.0, step=0.1, key='conv_ref_dx')
            ref_dx_m = ref_dx * 1e-6

            obs_conv_x = st.number_input('观测点 X (格点)', value=int(config.nx / 2),
                                         min_value=0, max_value=config.nx - 1, key='conv_obs_x')
            obs_conv_y = st.number_input('观测点 Y (格点)', value=int(config.ny / 2),
                                         min_value=0, max_value=config.ny - 1, key='conv_obs_y')

        with col2:
            levels = st.multiselect('网格密度级别 (相对于参考密度的倍数)',
                                    ['1x', '2x', '4x', '8x'],
                                    default=['1x', '2x', '4x'],
                                    key='conv_levels')

            st.info('💡 倍数越高，网格越密，计算时间越长')

        if st.button('▶️ 运行收敛性分析', key='run_convergence', type='primary'):
            if not levels:
                st.error('请至少选择一个网格密度级别')
            else:
                with st.spinner('正在进行收敛性分析...'):
                    level_factors = []
                    for lvl in levels:
                        if lvl == '1x':
                            level_factors.append(1)
                        elif lvl == '2x':
                            level_factors.append(2)
                        elif lvl == '4x':
                            level_factors.append(4)
                        elif lvl == '8x':
                            level_factors.append(8)
                    level_factors.sort()

                    progress_bar = st.progress(0)
                    convergence_results = []

                    for idx, factor_lvl in enumerate(level_factors):
                        dx_conv = ref_dx_m / factor_lvl
                        config_conv = copy.deepcopy(st.session_state.config)
                        config_conv.dx = dx_conv
                        config_conv.dy = dx_conv
                        config_conv.observation_points = [(int(obs_conv_x * factor_lvl),
                                                           int(obs_conv_y * factor_lvl))]

                        mat_lib_conv = copy.deepcopy(st.session_state.material_lib)
                        struct_conv = copy.deepcopy(st.session_state.structure_mgr)
                        src_conv = copy.deepcopy(st.session_state.source_mgr)
                        bc_conv = copy.deepcopy(st.session_state.boundary)

                        fdtd_conv = FDTD2D(config_conv, mat_lib_conv, struct_conv, src_conv, bc_conv)
                        result_conv = fdtd_conv.run()

                        obs_pt = (int(obs_conv_x * factor_lvl), int(obs_conv_y * factor_lvl))
                        obs_data = result_conv.observation_data.get(obs_pt, np.array([]))

                        convergence_results.append({
                            'factor': factor_lvl,
                            'dx': dx_conv,
                            'nx': config_conv.nx,
                            'ny': config_conv.ny,
                            'time_points': result_conv.observation_times,
                            'field_data': obs_data,
                            'computation_time': result_conv.computation_time
                        })

                        progress_bar.progress((idx + 1) / len(level_factors))

                    l2_errors = []
                    for i in range(len(convergence_results) - 1):
                        coarse = convergence_results[i]
                        fine = convergence_results[i + 1]

                        n_coarse = len(coarse['field_data'])
                        n_fine = len(fine['field_data'])

                        if n_coarse > 0 and n_fine > 0:
                            step_ratio = fine['factor'] // coarse['factor']
                            coarse_data = coarse['field_data']
                            fine_sampled = fine['field_data'][::step_ratio]

                            min_len = min(len(coarse_data), len(fine_sampled))
                            coarse_data = coarse_data[:min_len]
                            fine_sampled = fine_sampled[:min_len]

                            l2_error = np.sqrt(np.sum((coarse_data - fine_sampled) ** 2)) / \
                                       (np.sqrt(np.sum(fine_sampled ** 2)) + 1e-30)
                            l2_errors.append({
                                'from_factor': coarse['factor'],
                                'to_factor': fine['factor'],
                                'error': l2_error
                            })

                    st.session_state.convergence_result = {
                        'results': convergence_results,
                        'l2_errors': l2_errors,
                        'obs_point': (obs_conv_x, obs_conv_y)
                    }

                    progress_bar.empty()
                    st.success('收敛性分析完成!')

        if st.session_state.convergence_result is not None:
            conv_data = st.session_state.convergence_result
            conv_results = conv_data['results']
            l2_errors = conv_data['l2_errors']

            st.divider()
            st.subheader('时域波形对比')

            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['b', 'g', 'r', 'm']
            for i, res in enumerate(conv_results):
                color = colors[i % len(colors)]
                ax.plot(res['time_points'] * 1e9, res['field_data'],
                        f'{color}-', linewidth=1,
                        label=f"{res['factor']}x (dx={res['dx']*1e6:.2f}um, {res['nx']}×{res['ny']})")
            ax.set_xlabel('Time (ns)')
            ax.set_ylabel('Ez (V/m)')
            ax.set_title(f"不同网格密度下观测点 {conv_data['obs_point']} 的时域波形")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.divider()
            st.subheader('收敛曲线 (L2相对误差)')

            if len(l2_errors) > 0:
                fig, ax = plt.subplots(figsize=(8, 5))
                x_labels = [f"{err['from_factor']}x→{err['to_factor']}x" for err in l2_errors]
                y_errors = [err['error'] for err in l2_errors]

                ax.bar(x_labels, y_errors, color='steelblue', alpha=0.7)
                ax.set_ylabel('L2 相对误差')
                ax.set_title('相邻网格密度间的相对误差')
                ax.grid(True, alpha=0.3, axis='y')

                for i, err in enumerate(y_errors):
                    ax.text(i, err * 1.05, f'{err:.2e}', ha='center', va='bottom')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric('最小误差', f'{min(y_errors):.2e}')
                with col2:
                    st.metric('最大误差', f'{max(y_errors):.2e}')
            else:
                st.info('请选择至少两个网格密度级别以计算收敛误差')

            st.divider()
            st.subheader('计算信息')
            info_cols = st.columns(len(conv_results))
            for i, res in enumerate(conv_results):
                with info_cols[i]:
                    st.metric(f"{res['factor']}x 密度", f"{res['nx']}×{res['ny']}")
                    st.metric('计算耗时', f"{res['computation_time']:.2f}s")
                    st.caption(f"dx={res['dx']*1e6:.3f}um")

    st.divider()

    with st.expander('📤 数据导出', expanded=False):
        col_exp1, col_exp2, col_exp3, col_exp4 = st.columns(4)

        with col_exp1:
            field_csv = export_all_fields_csv(result)
            st.download_button(
                '📊 场数据 CSV',
                data=field_csv,
                file_name='field_data.csv',
                mime='text/csv',
                use_container_width=True
            )

        with col_exp2:
            obs_csv = export_observation_csv(result)
            st.download_button(
                '📈 观测点 CSV',
                data=obs_csv,
                file_name='observation_data.csv',
                mime='text/csv',
                use_container_width=True
            )

        with col_exp3:
            energy_csv = export_energy_csv(result, config.dx, config.dy)
            st.download_button(
                '⚡ 能量数据 CSV',
                data=energy_csv,
                file_name='energy_data.csv',
                mime='text/csv',
                use_container_width=True
            )

        with col_exp4:
            if st.button('🎬 生成动画 GIF', key='gen_gif', use_container_width=True):
                with st.spinner('正在生成 GIF...'):
                    try:
                        color_grid = getattr(st.session_state, 'color_grid', None)
                        frames = generate_animation_frames(
                            result.ez_frames[:100],
                            dx=config.dx, dy=config.dy, unit=config.unit,
                            color_grid=color_grid
                        )
                        gif_buf = io.BytesIO()
                        if frames:
                            frames[0].save(
                                gif_buf,
                                format='GIF',
                                save_all=True,
                                append_images=frames[1:],
                                duration=100,
                                loop=0
                            )
                            gif_buf.seek(0)
                            st.session_state.gif_data = gif_buf.getvalue()
                            st.success('GIF生成成功!')
                    except Exception as e:
                        st.error(f'GIF生成失败: {e}')

            if hasattr(st.session_state, 'gif_data') and st.session_state.gif_data:
                st.download_button(
                    '📥 下载 GIF',
                    data=st.session_state.gif_data,
                    file_name='fdtd_animation.gif',
                    mime='image/gif',
                    use_container_width=True
                )

st.divider()
st.caption('FDTD 电磁仿真工具 | 基于 Yee 网格的 2D TM 模式仿真')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Tuple, List
import io
from PIL import Image
import platform

plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei',
    'Noto Sans CJK SC', 'Arial Unicode MS', 'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False

EPS0 = 8.854e-12
MU0 = 4 * np.pi * 1e-7


def _ez_colormap():
    colors = [
        (0.0, (0.0, 0.0, 1.0)),
        (0.5, (1.0, 1.0, 1.0)),
        (1.0, (1.0, 0.0, 0.0)),
    ]
    return LinearSegmentedColormap.from_list('ez_bwr', colors, N=256)


def plot_field_heatmap(ez: np.ndarray, title: str = 'Ez Field',
                       dx: float = 1e-6, dy: float = 1e-6,
                       unit: str = 'um',
                       color_grid: Optional[np.ndarray] = None,
                       vmin: Optional[float] = None,
                       vmax: Optional[float] = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))

    scale = 1e6 if unit == 'um' else 1e3
    extent = [0, ez.shape[1] * dy * scale, 0, ez.shape[0] * dx * scale]

    if color_grid is not None:
        ax.imshow(color_grid.transpose(1, 0, 2), extent=extent, origin='lower', alpha=0.3)

    cmap = _ez_colormap()
    if vmin is None:
        vmax_val = np.max(np.abs(ez)) if np.max(np.abs(ez)) > 0 else 1.0
        vmin = -vmax_val
        vmax = vmax_val

    im = ax.imshow(ez.T, extent=extent, origin='lower', cmap=cmap,
                   vmin=vmin, vmax=vmax, alpha=0.8)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Ez (V/m)')

    ax.set_xlabel(f'Y ({unit})')
    ax.set_ylabel(f'X ({unit})')
    ax.set_title(title)
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig


def plot_time_waveform(time_points: np.ndarray, field_data: np.ndarray,
                       point_label: str = 'Observation Point',
                       title: str = 'Time Domain Waveform') -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(time_points * 1e9, field_data, 'b-', linewidth=1)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Ez (V/m)')
    ax.set_title(f'{title} - {point_label}')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_frequency_spectrum(time_points: np.ndarray, field_data: np.ndarray,
                            title: str = 'Frequency Spectrum') -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))

    n = len(field_data)
    dt = time_points[1] - time_points[0] if len(time_points) > 1 else 1e-12
    freq = np.fft.fftfreq(n, dt)
    spectrum = np.abs(np.fft.fft(field_data))
    spectrum_db = 20 * np.log10(spectrum + 1e-30)

    positive_mask = freq >= 0
    ax.plot(freq[positive_mask] / 1e9, spectrum_db[positive_mask], 'r-', linewidth=1)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_far_field_polar(angles: np.ndarray, far_field: np.ndarray,
                         title: str = 'Far Field Pattern') -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})

    normalized = far_field / (np.max(far_field) + 1e-30)
    ax.plot(angles, normalized, 'b-', linewidth=1.5)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(-1)
    ax.set_title(title, y=1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_rcs(angles: np.ndarray, rcs_db: np.ndarray,
             title: str = 'Bistatic RCS') -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(np.degrees(angles), rcs_db, 'r-', linewidth=1)
    ax.set_xlabel('Angle (deg)')
    ax.set_ylabel('RCS (dBsm)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 360)

    plt.tight_layout()
    return fig


def plot_energy_density(energy_density: np.ndarray,
                        dx: float = 1e-6, dy: float = 1e-6,
                        unit: str = 'um',
                        title: str = 'Energy Density') -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))

    scale = 1e6 if unit == 'um' else 1e3
    extent = [0, energy_density.shape[1] * dy * scale, 0, energy_density.shape[0] * dx * scale]

    im = ax.imshow(energy_density.T, extent=extent, origin='lower',
                   cmap='hot', vmin=0)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Energy Density (J/m^3)')

    ax.set_xlabel(f'Y ({unit})')
    ax.set_ylabel(f'X ({unit})')
    ax.set_title(title)
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig


def fig_to_image(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return img


def fig_to_bytes(fig: plt.Figure, format: str = 'png') -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=100, bbox_inches='tight')
    buf.seek(0)
    return buf.read()


def generate_animation_frames(ez_frames: np.ndarray,
                              dx: float = 1e-6, dy: float = 1e-6,
                              unit: str = 'um',
                              color_grid: Optional[np.ndarray] = None) -> List[Image.Image]:
    vmax = np.max(np.abs(ez_frames)) if len(ez_frames) > 0 else 1.0
    if vmax == 0:
        vmax = 1.0

    frames = []
    for i, ez in enumerate(ez_frames):
        fig = plot_field_heatmap(
            ez,
            title=f'Ez Field - Frame {i}',
            dx=dx, dy=dy, unit=unit,
            color_grid=color_grid,
            vmin=-vmax, vmax=vmax
        )
        img = fig_to_image(fig)
        frames.append(img)
        plt.close(fig)

    return frames


def save_gif(frames: List[Image.Image], output_path: str,
             duration: int = 100, loop: int = 0):
    if not frames:
        return
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop
    )


def plot_parametric_sweep(param_values: np.ndarray, metric_values: np.ndarray,
                          param_name: str = 'Parameter',
                          metric_name: str = 'Metric',
                          title: str = 'Parametric Sweep') -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(param_values, metric_values, 'bo-', linewidth=1.5, markersize=4)
    ax.set_xlabel(param_name)
    ax.set_ylabel(metric_name)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_sparam_magnitude(s_result, frequency_unit: str = 'GHz',
                          y_min: float = -60.0, y_max: float = 5.0,
                          show_invalid: bool = True) -> plt.Figure:
    from .sparam import SParameterResult
    fig, ax = plt.subplots(figsize=(10, 6))
    freq_scale = {
        'Hz': 1.0, 'kHz': 1e3, 'MHz': 1e6, 'GHz': 1e9, 'THz': 1e12
    }[frequency_unit]
    colors = plt.cm.tab10(np.linspace(0, 1, s_result.num_ports ** 2))
    color_idx = 0
    frequencies = s_result.frequencies / freq_scale
    valid_mask = s_result.valid_bandwidth_mask
    passivity_mask = s_result.passivity_violation_mask
    for i in range(s_result.num_ports):
        for j in range(s_result.num_ports):
            s_ij = s_result.s_matrix[i, j, :]
            mag_db = 20 * np.log10(np.abs(s_ij) + 1e-30)
            color = colors[color_idx]
            ax.plot(frequencies[valid_mask], mag_db[valid_mask],
                    color=color, linewidth=1.5, label=f'S{i+1}{j+1}')
            if show_invalid and np.any(~valid_mask):
                ax.plot(frequencies[~valid_mask], mag_db[~valid_mask],
                        color=color, linewidth=1.0, linestyle='--', alpha=0.5)
            if np.any(passivity_mask):
                ax.scatter(frequencies[passivity_mask], mag_db[passivity_mask],
                           color='red', s=20, marker='o', zorder=5,
                           label='_nolegend_' if color_idx > 0 else 'Passivity violation')
            color_idx += 1
    ax.set_xlabel(f'Frequency ({frequency_unit})')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('S-Parameters - Magnitude')
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    return fig


def plot_sparam_phase(s_result, frequency_unit: str = 'GHz',
                      unwrap: bool = True, show_invalid: bool = True) -> plt.Figure:
    from .sparam import SParameterResult
    fig, ax = plt.subplots(figsize=(10, 6))
    freq_scale = {
        'Hz': 1.0, 'kHz': 1e3, 'MHz': 1e6, 'GHz': 1e9, 'THz': 1e12
    }[frequency_unit]
    colors = plt.cm.tab10(np.linspace(0, 1, s_result.num_ports ** 2))
    color_idx = 0
    frequencies = s_result.frequencies / freq_scale
    valid_mask = s_result.valid_bandwidth_mask
    for i in range(s_result.num_ports):
        for j in range(s_result.num_ports):
            s_ij = s_result.s_matrix[i, j, :]
            phase = np.angle(s_ij, deg=True)
            if unwrap:
                phase = np.unwrap(phase, period=360)
            color = colors[color_idx]
            ax.plot(frequencies[valid_mask], phase[valid_mask],
                    color=color, linewidth=1.5, label=f'S{i+1}{j+1}')
            if show_invalid and np.any(~valid_mask):
                ax.plot(frequencies[~valid_mask], phase[~valid_mask],
                        color=color, linewidth=1.0, linestyle='--', alpha=0.5)
            color_idx += 1
    ax.set_xlabel(f'Frequency ({frequency_unit})')
    ax.set_ylabel('Phase (degrees)')
    ax.set_title('S-Parameters - Phase')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    return fig


def plot_smith_chart(s_result, port_indices: Optional[List[int]] = None,
                     num_markers: int = 10) -> plt.Figure:
    from .sparam import SParameterResult
    fig, ax = plt.subplots(figsize=(8, 8))
    if port_indices is None:
        port_indices = [0] if s_result.num_ports >= 1 else []
        if s_result.num_ports >= 2:
            port_indices.append(1)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=0.8, alpha=0.8)
    for r_val in [0.2, 0.4, 0.6, 0.8]:
        ax.plot(r_val * np.cos(theta), r_val * np.sin(theta),
                'k-', linewidth=0.3, alpha=0.3)
    for r_norm in [0.2, 0.5, 1.0, 2.0, 5.0]:
        center_x = r_norm / (1 + r_norm)
        radius = 1 / (1 + r_norm)
        circle_pts = center_x + radius * np.exp(1j * theta)
        ax.plot(np.real(circle_pts), np.imag(circle_pts),
                'k-', linewidth=0.3, alpha=0.3)
    for x_norm in [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]:
        center_x = 1.0
        center_y = 1.0 / x_norm if x_norm != 0 else 1e10
        radius = np.abs(1.0 / x_norm) if x_norm != 0 else 1e10
        if radius < 5:
            circle_pts = center_x + 1j * center_y + radius * np.exp(1j * theta)
            inside_mask = np.abs(circle_pts) <= 1.01
            ax.plot(np.real(circle_pts[inside_mask]), np.imag(circle_pts[inside_mask]),
                    'k-', linewidth=0.3, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.3, alpha=0.5)
    ax.axvline(x=0, color='k', linewidth=0.3, alpha=0.5)
    ax.text(0.02, 0, '0', fontsize=7, ha='left', va='center')
    ax.text(0.5, 0, '0.5', fontsize=7, ha='center', va='center')
    ax.text(1, 0, '∞', fontsize=10, ha='right', va='center')
    ax.text(0.95, 0.5, 'j1', fontsize=7, ha='right', va='center')
    ax.text(0.95, -0.5, '-j1', fontsize=7, ha='right', va='center')
    ax.text(0.5, 0.866, 'j2', fontsize=7, ha='center', va='bottom')
    ax.text(0.5, -0.866, '-j2', fontsize=7, ha='center', va='top')
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    for idx, port_idx in enumerate(port_indices):
        if port_idx >= s_result.num_ports:
            continue
        s_ii = s_result.s_matrix[port_idx, port_idx, :]
        valid_mask = s_result.valid_bandwidth_mask
        s_valid = s_ii[valid_mask]
        freq_valid = s_result.frequencies[valid_mask]
        if len(s_valid) > 0:
            gamma_real = np.real(s_valid)
            gamma_imag = np.imag(s_valid)
            ax.plot(gamma_real, gamma_imag,
                    color=colors[idx % len(colors)],
                    linewidth=1.5, label=f'S{port_idx+1}{port_idx+1}')
            if num_markers > 0 and len(freq_valid) > num_markers:
                marker_step = max(1, len(freq_valid) // num_markers)
                marker_indices = np.arange(0, len(freq_valid), marker_step)
                for mi in marker_indices:
                    freq_ghz = freq_valid[mi] / 1e9
                    label = f'{freq_ghz:.1f}'
                    ax.annotate(label,
                                xy=(gamma_real[mi], gamma_imag[mi]),
                                fontsize=7,
                                textcoords='offset points',
                                xytext=(6, 6),
                                bbox=dict(boxstyle='round,pad=0.2', fc='white',
                                          alpha=0.8, ec='none'))
                    ax.plot([gamma_real[mi]], [gamma_imag[mi]], 'o',
                            color=colors[idx % len(colors)], markersize=5,
                            markeredgecolor='white', markeredgewidth=0.5)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title('Smith Chart - Reflection Coefficients', fontsize=12, y=1.02)
    ax.grid(True, alpha=0.1)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    plt.tight_layout()
    return fig


def plot_tdr(distance: np.ndarray, impedance: np.ndarray,
             distance_unit: str = 'mm',
             z0: float = 50.0, title: str = 'TDR Response') -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    dist_scale = {
        'm': 1.0, 'cm': 1e-2, 'mm': 1e-3, 'um': 1e-6
    }[distance_unit]
    ax.plot(distance / dist_scale, impedance, 'b-', linewidth=1.5)
    ax.axhline(y=z0, color='r', linestyle='--', linewidth=1, alpha=0.7, label=f'Z0 = {z0} Ohm')
    ax.set_xlabel(f'Distance ({distance_unit})')
    ax.set_ylabel('Impedance (Ohm)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(max(0, z0 * 0.5), min(z0 * 2, np.max(impedance) * 1.1))
    plt.tight_layout()
    return fig


def plot_port_time_signals(s_result, port_idx: int = 0,
                           time_unit: str = 'ns') -> plt.Figure:
    from .sparam import SParameterResult
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    time_scale = {
        's': 1.0, 'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9, 'ps': 1e-12
    }[time_unit]
    excite_key = f'excite_{port_idx}'
    if excite_key not in s_result.time_signals:
        return fig
    port_data = s_result.time_signals[excite_key]
    colors = plt.cm.tab10(np.linspace(0, 1, s_result.num_ports))
    for p_idx in range(s_result.num_ports):
        if p_idx in port_data:
            data = port_data[p_idx]
            time_arr = data['time'] / time_scale
            axes[0].plot(time_arr, data['v_plus'], color=colors[p_idx],
                         linewidth=1, label=f'Port {p_idx+1} V+')
            axes[1].plot(time_arr, data['v_minus'], color=colors[p_idx],
                         linewidth=1, label=f'Port {p_idx+1} V-')
    axes[0].set_ylabel('Incident Wave V+ (V)')
    axes[0].set_title(f'Time Domain Signals - Excitation at Port {port_idx + 1}')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[1].set_xlabel(f'Time ({time_unit})')
    axes[1].set_ylabel('Reflected Wave V- (V)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    return fig


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

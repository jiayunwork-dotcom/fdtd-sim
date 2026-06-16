import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import json


@dataclass
class Material:
    name: str
    epsilon_r: float
    sigma: float
    mu_r: float = 1.0
    color: str = '#888888'

    def to_dict(self):
        return {
            'name': self.name,
            'epsilon_r': self.epsilon_r,
            'sigma': self.sigma,
            'mu_r': self.mu_r,
            'color': self.color
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data['name'],
            epsilon_r=data['epsilon_r'],
            sigma=data['sigma'],
            mu_r=data.get('mu_r', 1.0),
            color=data.get('color', '#888888')
        )


class MaterialLibrary:
    def __init__(self):
        self.materials = {}
        self._init_default_materials()

    def _init_default_materials(self):
        self.add_material(Material('Air', 1.0, 0.0, color='#E0E0E0'))
        self.add_material(Material('Glass', 4.5, 0.0, color='#A0D8EF'))
        self.add_material(Material('Silicon', 11.7, 0.001, color='#C0C0C0'))
        self.add_material(Material('Copper', 1.0, 5.96e7, color='#B87333'))
        self.add_material(Material('Water', 80.0, 5.5e-4, color='#4169E1'))
        self.add_material(Material('PEC', 1.0, 1e20, color='#FFD700'))

    def add_material(self, material: Material):
        self.materials[material.name] = material

    def get_material(self, name: str) -> Material:
        return self.materials.get(name, self.materials['Air'])

    def list_materials(self) -> List[str]:
        return list(self.materials.keys())

    def to_dict(self):
        return {name: mat.to_dict() for name, mat in self.materials.items()}

    @classmethod
    def from_dict(cls, data):
        lib = cls()
        lib.materials = {}
        for name, mat_data in data.items():
            lib.materials[name] = Material.from_dict(mat_data)
        return lib


@dataclass
class Structure:
    shape_type: str
    material_name: str
    params: dict
    is_pec: bool = False

    def to_dict(self):
        return {
            'shape_type': self.shape_type,
            'material_name': self.material_name,
            'params': self.params,
            'is_pec': self.is_pec
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            shape_type=data['shape_type'],
            material_name=data['material_name'],
            params=data['params'],
            is_pec=data.get('is_pec', False)
        )

    def contains(self, x: float, y: float) -> bool:
        if self.shape_type == 'rectangle':
            x0, y0 = self.params['x0'], self.params['y0']
            w, h = self.params['width'], self.params['height']
            return x0 <= x <= x0 + w and y0 <= y <= y0 + h
        elif self.shape_type == 'circle':
            cx, cy = self.params['cx'], self.params['cy']
            r = self.params['radius']
            return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2
        elif self.shape_type == 'line':
            x0, y0 = self.params['x0'], self.params['y0']
            x1, y1 = self.params['x1'], self.params['y1']
            thickness = self.params.get('thickness', 1e-6)
            dx = x1 - x0
            dy = y1 - y0
            length = np.sqrt(dx ** 2 + dy ** 2)
            if length == 0:
                return abs(x - x0) < thickness and abs(y - y0) < thickness
            t = max(0, min(1, ((x - x0) * dx + (y - y0) * dy) / (length ** 2)))
            px = x0 + t * dx
            py = y0 + t * dy
            return np.sqrt((x - px) ** 2 + (y - py) ** 2) < thickness
        return False


class StructureManager:
    def __init__(self):
        self.structures: List[Structure] = []

    def add_structure(self, structure: Structure):
        self.structures.append(structure)

    def remove_structure(self, index: int):
        if 0 <= index < len(self.structures):
            self.structures.pop(index)

    def clear(self):
        self.structures = []

    def generate_material_grid(self, nx: int, ny: int, dx: float, dy: float,
                               material_lib: MaterialLibrary) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        epsilon_r = np.ones((nx, ny))
        sigma = np.zeros((nx, ny))
        mu_r = np.ones((nx, ny))
        color_grid = np.full((nx, ny, 3), 224, dtype=np.uint8)

        x_grid = np.arange(nx) * dx
        y_grid = np.arange(ny) * dy
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')

        for structure in self.structures:
            mask = self._structure_mask(structure, X, Y, nx, ny, dx, dy)
            if np.any(mask):
                mat = material_lib.get_material(structure.material_name)
                if structure.is_pec or mat.name == 'PEC':
                    epsilon_r[mask] = 1.0
                    sigma[mask] = 1e20
                    mu_r[mask] = 1.0
                else:
                    epsilon_r[mask] = mat.epsilon_r
                    sigma[mask] = mat.sigma
                    mu_r[mask] = mat.mu_r
                color = mat.color.lstrip('#')
                r = int(color[0:2], 16)
                g = int(color[2:4], 16)
                b = int(color[4:6], 16)
                color_grid[mask, 0] = r
                color_grid[mask, 1] = g
                color_grid[mask, 2] = b

        return epsilon_r, sigma, mu_r, color_grid

    def _structure_mask(self, structure: Structure, X: np.ndarray, Y: np.ndarray,
                        nx: int, ny: int, dx: float, dy: float) -> np.ndarray:
        if structure.shape_type == 'rectangle':
            x0, y0 = structure.params['x0'], structure.params['y0']
            w, h = structure.params['width'], structure.params['height']
            return (X >= x0) & (X <= x0 + w) & (Y >= y0) & (Y <= y0 + h)
        elif structure.shape_type == 'circle':
            cx, cy = structure.params['cx'], structure.params['cy']
            r = structure.params['radius']
            return (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
        elif structure.shape_type == 'line':
            x0, y0 = structure.params['x0'], structure.params['y0']
            x1, y1 = structure.params['x1'], structure.params['y1']
            thickness = structure.params.get('thickness', 1e-6)
            dx_line = x1 - x0
            dy_line = y1 - y0
            length = np.sqrt(dx_line ** 2 + dy_line ** 2)
            if length == 0:
                return (np.abs(X - x0) < thickness) & (np.abs(Y - y0) < thickness)
            t = ((X - x0) * dx_line + (Y - y0) * dy_line) / (length ** 2)
            t = np.clip(t, 0, 1)
            px = x0 + t * dx_line
            py = y0 + t * dy_line
            return np.sqrt((X - px) ** 2 + (Y - py) ** 2) < thickness
        return np.zeros_like(X, dtype=bool)

    def to_dict(self):
        return [s.to_dict() for s in self.structures]

    @classmethod
    def from_dict(cls, data):
        mgr = cls()
        mgr.structures = [Structure.from_dict(s) for s in data]
        return mgr

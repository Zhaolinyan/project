#!/usr/bin/env python3
"""
Cylinder3D inference engine subprocess (independent of ROS2)

Features:
  - Read point cloud data from stdin (numpy binary format, explicit little-endian)
  - Execute Cylinder3D semantic segmentation inference
  - Write labels to stdout (explicit little-endian)

Communication protocol:
  stdin:  [n_pts(8B int64 '<q')][xyz(n*12B f32)][intensity(n*4B f32)]
  stdout: [n_labels(8B int64 '<q')][labels(n*4B uint32 '<u4')]

Key fixes (vs old version):
  1. struct.pack uses '<q' explicit little-endian byte order
  2. Suppress torch/spconv FutureWarning to keep stderr clean
  3. Read loop ensures complete data reception from stdin

This script runs in the cylinder3d conda environment (Python 3.8 + PyTorch + spconv)
"""

import os
import sys
import struct
import argparse
import warnings
import numpy as np

# Suppress torch/spconv FutureWarning to keep stderr clean (prevent deadlock)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*spconv.*')
warnings.filterwarnings('ignore', message='.*torch.*')

# ====== Locate Cylinder3D root directory ======
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _find_cylinder3d_root():
    """Multi-strategy locate Cylinder3D directory"""
    candidates = [
        os.environ.get('CYLINDER3D_ROOT', ''),
        # Installed: install/dsw_lio_sam/lib/dsw_lio_sam -> src/... (up 3 levels)
        os.path.join(_SCRIPT_DIR, '..', '..', '..', 'src', 'dsw_lio_sam', 'Cylinder3D'),
        # Source direct run: semantic_bridge/ -> ../Cylinder3D
        os.path.join(_SCRIPT_DIR, '..', 'Cylinder3D'),
    ]
    for c in candidates:
        # builder is a Python package directory, not a single builder.py file
        if os.path.isdir(os.path.join(c, 'builder')) and \
           os.path.isfile(os.path.join(c, 'builder', '__init__.py')):
            return os.path.abspath(c)
    raise RuntimeError(
        f'Cylinder3D not found! Searched:\n' +
        '\n'.join(f'  - {c}' for c in candidates))

CYLINDER3D_ROOT = _find_cylinder3d_root()
sys.path.insert(0, CYLINDER3D_ROOT)

import torch
import yaml
from builder import model_builder
from config.config import load_config_data
from utils.load_save_util import load_checkpoint
from utils.inference_profile import resolve_inference_profile


def read_exact(fd, n_bytes):
    """Loop read from file descriptor until n_bytes collected or EOF"""
    buf = b''
    while len(buf) < n_bytes:
        remaining = n_bytes - len(buf)
        chunk = fd.read(min(remaining, 65536))
        if not chunk:
            if len(buf) == 0:
                return None
            return None
        buf += chunk
    return buf


class Cylinder3DInferenceEngine:
    """Cylinder3D inference engine"""

    def __init__(self, config_path='config/newer_college.yaml',
                 device='cuda:0',
                 z_min=None, z_max=None,
                 rho_max=None):
        project_root = CYLINDER3D_ROOT

        # ---- Load model ----
        config_abs = config_path if os.path.isabs(config_path) else os.path.join(project_root, config_path)
        configs = load_config_data(config_abs)
        model_config = configs['model_params']
        train_hypers = configs['train_params']
        dataset_config = configs['dataset_params']
        profile = resolve_inference_profile(project_root, train_hypers, dataset_config)

        if not os.path.isabs(dataset_config['label_mapping']):
            dataset_config['label_mapping'] = os.path.join(
                project_root, dataset_config['label_mapping'])

        model_load_path = profile['checkpoint']

        self.model = model_builder.build(model_config)
        self.model = load_checkpoint(model_load_path, self.model)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self.grid_size = np.asarray(model_config['output_shape'])
        self.num_class = model_config['num_class']

        # ---- Voxel space parameters ----
        min_volume = np.asarray(profile['min_volume_space'], dtype=np.float32)
        max_volume = np.asarray(profile['max_volume_space'], dtype=np.float32)
        if z_min is not None:
            min_volume[2] = z_min
        if z_max is not None:
            max_volume[2] = z_max
        if rho_max is not None:
            max_volume[0] = rho_max
        self.min_volume_space = min_volume
        self.max_volume_space = max_volume

        # ---- Label mapping ----
        with open(dataset_config['label_mapping'], 'r') as f:
            semkittiyaml = yaml.safe_load(f)
        self.inv_learning_map = semkittiyaml['learning_map_inv']

        # KITTI -> DSW mapping
        kitti_to_dsw = {
            0: 0, 10: 1, 11: 5, 13: 3, 15: 6, 16: 3, 18: 2, 20: 2,
            30: 4, 31: 5, 32: 6, 40: 8, 44: 8, 48: 9, 49: 10, 50: 7,
            51: 13, 52: 14, 60: 9, 70: 11, 71: 11, 72: 10, 80: 12, 81: 12, 99: 0,
        }

        # Pre-build LUT
        max_learn_id = max(self.inv_learning_map.keys()) + 1
        self._inv_lut = np.zeros(max_learn_id, dtype=np.uint32)
        for learn_id, orig_id in self.inv_learning_map.items():
            self._inv_lut[learn_id] = orig_id

        max_kitti_id = max(max(kitti_to_dsw.keys()), max(self.inv_learning_map.values())) + 1
        self._dsw_lut = np.zeros(max_kitti_id, dtype=np.uint32)
        for kitti_id, dsw_id in kitti_to_dsw.items():
            self._dsw_lut[kitti_id] = dsw_id

    def _cart2polar(self, xyz):
        rho = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        return np.stack((rho, phi, xyz[:, 2]), axis=1)

    def _preprocess(self, xyz, intensity):
        """Preprocess point cloud"""
        xyz_pol = self._cart2polar(xyz)
        max_bound = self.max_volume_space
        min_bound = self.min_volume_space
        crop_range = max_bound - min_bound
        intervals = crop_range / (self.grid_size - 1)
        clipped = np.clip(xyz_pol, min_bound, max_bound)
        grid_ind = np.floor((clipped - min_bound) / intervals).astype(np.int64)
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_fea = np.concatenate(
            (return_xyz, xyz_pol, xyz[:, :2], intensity[..., np.newaxis]), axis=1)
        return grid_ind, return_fea

    def predict(self, xyz, intensity):
        """Execute inference, return DSW labels (uint32 numpy array)"""
        with torch.no_grad():
            grid_ind, pt_fea = self._preprocess(xyz, intensity)
            pt_fea_ten = [torch.from_numpy(pt_fea).float().to(self.device)]
            grid_ten = [torch.from_numpy(grid_ind).to(self.device)]

            predict_labels = self.model(pt_fea_ten, grid_ten, 1)
            predict_labels = torch.argmax(predict_labels, dim=1).cpu().numpy()
            point_labels = predict_labels[0, grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2]]

            kitti_ids = self._inv_lut[point_labels]
            dsw_labels = self._dsw_lut[kitti_ids]
            return dsw_labels.astype(np.uint32)


def main():
    parser = argparse.ArgumentParser(description='Cylinder3D inference engine')
    parser.add_argument('--config_path', type=str, default='config/newer_college.yaml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--z_min', type=float, default=None)
    parser.add_argument('--z_max', type=float, default=None)
    parser.add_argument('--rho_max', type=float, default=None)
    args = parser.parse_args()

    # Initialize engine (load model to GPU)
    engine = Cylinder3DInferenceEngine(
        config_path=args.config_path,
        device=args.device,
        z_min=args.z_min, z_max=args.z_max,
        rho_max=args.rho_max)

    # Signal ready - only write critical info to stderr (now DEVNULL on parent side anyway)
    # But keep this for debugging if stderr is redirected to a log file
    sys.stderr.write('[ENGINE] Model loaded, ready for inference\n')
    sys.stderr.flush()

    # ====== Main loop: read from stdin, inference, write to stdout ======
    while True:
        # 1. Read point count (8 bytes int64, explicit little-endian)
        raw_n = read_exact(sys.stdin.buffer, 8)
        if raw_n is None:
            break  # EOF (parent closed pipe)
        n_points = struct.unpack('<q', raw_n)[0]  # <<< FIX: explicit little-endian

        if n_points <= 0 or n_points > 2000000:  # Sanity check
            continue

        # 2. Read xyz (n*12 bytes float32)
        xyz_raw = read_exact(sys.stdin.buffer, n_points * 12)
        if xyz_raw is None:
            break
        xyz = np.frombuffer(xyz_raw, dtype='<f4').reshape(n_points, 3).copy()  # <<< FIX: explicit little-endian float32

        # 3. Read intensity (n*4 bytes float32)
        int_raw = read_exact(sys.stdin.buffer, n_points * 4)
        if int_raw is None:
            break
        intensity = np.frombuffer(int_raw, dtype='<f4').copy()  # <<< FIX: explicit little-endian float32

        # 4. Inference
        labels = engine.predict(xyz, intensity)

        # 5. Write back label count and labels (explicit little-endian)
        out_data = struct.pack('<q', len(labels)) + labels.astype('<u4').tobytes()  # <<< FIX: explicit little-endian
        sys.stdout.buffer.write(out_data)
        sys.stdout.buffer.flush()


if __name__ == '__main__':
    main()

"""Read/write EasyMocap camera files (intri.yml / extri.yml).

Mirrors easymocap/mytools/camera_utils.py so the files are byte-compatible
with what EasyMocap's own calibration tools produce:

  intri.yml: names, K_<cam> (3x3), dist_<cam> (1x5)
  extri.yml: names, R_<cam> (3x1 Rodrigues), Rot_<cam> (3x3), T_<cam> (3x1, meters)
"""
import os

import cv2
import numpy as np


def _write_yml(path, names, entries):
    """entries: list of (key, 2D np.ndarray) written as !!opencv-matrix."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lines = ['%YAML:1.0', '---', 'names:']
    lines += ['  - "{}"'.format(n) for n in names]
    for key, mat in entries:
        mat = np.asarray(mat, dtype=np.float64)
        lines += [
            '{}: !!opencv-matrix'.format(key),
            '  rows: {}'.format(mat.shape[0]),
            '  cols: {}'.format(mat.shape[1]),
            '  dt: d',
            '  data: [{}]'.format(', '.join('{:.6f}'.format(v) for v in mat.reshape(-1))),
        ]
    with open(path, 'w') as f:
        f.write('\r\n'.join(lines) + '\r\n')


def write_cameras(out_dir, cameras):
    """cameras: {name: {'K': 3x3, 'dist': 5, 'R': 3x3, 'T': 3 (meters)}}"""
    names = list(cameras.keys())
    intri, extri = [], []
    for n in names:
        cam = cameras[n]
        R = np.asarray(cam['R'], dtype=np.float64).reshape(3, 3)
        intri.append(('K_' + n, np.asarray(cam['K']).reshape(3, 3)))
        intri.append(('dist_' + n, np.asarray(cam['dist']).reshape(1, -1)[:, :5]))
        extri.append(('R_' + n, cv2.Rodrigues(R)[0]))
        extri.append(('Rot_' + n, R))
        extri.append(('T_' + n, np.asarray(cam['T']).reshape(3, 1)))
    _write_yml(os.path.join(out_dir, 'intri.yml'), names, intri)
    _write_yml(os.path.join(out_dir, 'extri.yml'), names, extri)


def _read_names(fs):
    node = fs.getNode('names')
    names = []
    for i in range(node.size()):
        val = node.at(i).string()
        if val == '':
            val = str(int(node.at(i).real()))
        names.append(val)
    return names


def read_cameras(root):
    """Read <root>/intri.yml and <root>/extri.yml into the dict format above."""
    intri = cv2.FileStorage(os.path.join(root, 'intri.yml'), cv2.FILE_STORAGE_READ)
    extri = cv2.FileStorage(os.path.join(root, 'extri.yml'), cv2.FILE_STORAGE_READ)
    cameras = {}
    for n in _read_names(intri):
        Rvec = extri.getNode('R_' + n).mat()
        cameras[n] = {
            'K': intri.getNode('K_' + n).mat(),
            'dist': intri.getNode('dist_' + n).mat(),
            'R': cv2.Rodrigues(Rvec)[0],
            'T': extri.getNode('T_' + n).mat(),
        }
    intri.release()
    extri.release()
    return cameras

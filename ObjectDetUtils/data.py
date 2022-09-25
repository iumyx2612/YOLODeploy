import numpy as np


def read_yolo_ann(ann_path):
    data = np.loadtxt(ann_path)
    if data.ndim == 1:
        data = data[None]
    cxs, cys, ws, hs = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    return cxs, cys, ws, hs

import numpy as np

import torch


def xyxy_to_cxcywh(xyxy):
    """
    Function to convert [x_min, y_min, x_max, y_max] to [cx, cy, w, h] box format
    Args:
        xyxy: bbox of format [x_min, y_min, x_max, y_max]
    Example:
        >>> xyxy = np.asarray([
        >>>     [100, 100, 200, 200]
        >>> ], dtype=np.int_)
        >>> cxcywh = xyxy_to_cxcywh(xyxy)
        >>> print(cxcywh)
    """
    if isinstance(xyxy, (list, tuple)):
        assert len(xyxy) == 4
        cX = round((xyxy[0] + xyxy[2]) / 2)
        cY = round((xyxy[1] + xyxy[3]) / 2)
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]
        return [cX, cY, w, h]
    elif isinstance(xyxy, np.ndarray):
        cX = np.round((xyxy[:, 0] + xyxy[:, 2]) / 2)
        cY = np.round((xyxy[:, 1] + xyxy[:, 3]) / 2)
        center = np.asarray([cX, cY], dtype=np.int_)
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        wh = np.asarray([w, h], dtype=np.int_)
        return np.vstack((center, wh)).T
    elif isinstance(xyxy, torch.Tensor):
        cX = torch.round((xyxy[:, 0] + xyxy[:, 2]) / 2)
        cY = torch.round((xyxy[:, 1] + xyxy[:, 3]) / 2)
        center = torch.vstack([cX, cY])
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        wh = torch.vstack([w, h])
        return torch.vstack((center, wh)).type(torch.int).T
    else:
        raise TypeError('Argument xyxy must be a list, tuple, numpy array or torch Tensor.')


def cxcywh_to_xyxy(xywh):
    """
    Function to convert [cx, cy, w, h] to [x_min, y_min, x_max, y_max] box format
    Args:
        xywh: bbox of format [cx, cy, w, h]
    Example:
        >>> xywh = np.asarray([
        >>>     [150, 150, 100, 100]
        >>> ], dtype=np.int_)
        >>> xyxy = cxcywh_to_xyxy(xywh)
        >>> print(xyxy)
    """
    if isinstance(xywh, (list, tuple)):
        assert len(xywh) == 4
        return [int(xywh[0] - xywh[2]/2),int(xywh[1] - xywh[3]/2),
                int(xywh[0] + xywh[2]/2), int(xywh[1] + xywh[3] / 2)]
    elif isinstance(xywh, np.ndarray):
        x1 = np.asarray(np.round(xywh[:, 0] - xywh[:, 2] / 2), dtype=np.int_)
        x2 = np.asarray(np.round(xywh[:, 0] + xywh[:, 2] / 2), dtype=np.int_)
        y1 = np.asarray(np.round(xywh[:, 1] - xywh[:, 3] / 2), dtype=np.int_)
        y2 = np.asarray(np.round(xywh[:, 1] + xywh[:, 3] / 2), dtype=np.int_)
        return np.vstack((x1, y1, x2, y2)).T
    elif isinstance(xywh, torch.Tensor):
        x1 = torch.round(xywh[:, 0] - xywh[:, 2] / 2)
        x2 = torch.round(xywh[:, 0] + xywh[:, 2] / 2)
        y1 = torch.round(xywh[:, 1] - xywh[:, 3] / 2)
        y2 = torch.round(xywh[:, 1] + xywh[:, 3] / 2)
        return torch.vstack((x1, y1, x2, y2)).type(torch.int).T


def xyxy_to_xywh(xyxy):
    """
    Function to convert [x_min, y_min, x_max, y_max] to [x_min, y_min, w, h] box format
    Args:
        xyxy: bbox of format [x_min, y_min, x_max, y_max]
    Example:
        >>> xyxy = np.asarray([
        >>>     [100, 100, 200, 200]
        >>> ], dtype=np.int_)
        >>> xywh = xyxy_to_xywh(xyxy)
        >>> print(xywh)
    """
    if isinstance(xyxy, (list, tuple)):
        assert len(xyxy) == 4
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]
        return [xyxy[0], xyxy[1], w, h]
    elif isinstance(xyxy, np.ndarray):
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        return np.vstack((xyxy[:, 0], xyxy[:, 1], w, h)).T
    elif isinstance(xyxy, torch.Tensor):
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        return torch.vstack((xyxy[:, 0], xyxy[:, 1], w, h)).type(torch.int).T


def xywh_to_xyxy(xywh):
    """
    Function to convert [x_min, y_min, w, h] to [x_min, y_min, x_max, x_max] box format
    Args:
        xyxy: bbox of format [x_min, y_min, w, h]
    Example:
        >>> xywh = np.asarray([
        >>>     [100, 100, 100, 100]
        >>> ], dtype=np.int_)
        >>> xyxt = xyxy_to_xywh(xywh)
        >>> print(xyxy)
    """
    if isinstance(xywh, (list, tuple)):
        assert len(xywh) == 4
        x2 = xywh[0] + xywh[2]
        y2 = xywh[1] + xywh[3]
        return [xywh[0], xywh[1], x2, y2]
    elif isinstance(xywh, np.ndarray):
        x2 = xywh[:, 0] + xywh[:, 2]
        y2 = xywh[:, 1] + xywh[:, 3]
        return np.vstack((xywh[:, 0], xywh[:, 1], x2, y2)).T
    elif isinstance(xywh, torch.Tensor):
        x2 = xywh[:, 0] + xywh[:, 2]
        y2 = xywh[:, 1] + xywh[:, 3]
        return torch.vstack((xywh[:, 0], xywh[:, 1], x2, y2)).type(torch.int).T


def scale_box(img, box):
    """ Scale normalized box w.r.t to image height and width
    Args:
        img: image of type numpy array or torch Tensor
        box: box of format xyxy or cxcywh (haven't tested with xywh)
    Example:
        >>> image = np.zeros((300, 300, 3))
        >>> norm_xyxy = np.asarray([
        >>>     [0.333, 0.333, 0.667, 0.667]
        >>> ])
        >>> xyxy = scale_box(image, norm_xyxy)
        >>> print(xyxy)
        [[100 100 200 200]]
    """
    if isinstance(box, (tuple, list)):
        h, w = img.shape[:2]
        nb1 = round(box[0] * w)
        nb2 = round(box[1] * h)
        nb3 = round(box[2] * w)
        nb4 = round(box[3] * h)
        return [nb1, nb2, nb3, nb4]
    if isinstance(box, np.ndarray):
        h, w = img.shape[:2] # img shape (h, w, c)
        nb1 = np.round(box[:, 0] * w)
        nb2 = np.round(box[:, 1] * h)
        nb3 = np.round(box[:, 2] * w)
        nb4 = np.round(box[:, 3] * h)
        return np.vstack((nb1, nb2, nb3, nb4)).astype(np.int_).T
    elif isinstance(box, torch.Tensor):
        if isinstance(img, torch.Tensor):
            w, h = img.shape[2:]  # img shape (batch, c, w, h)
        else:
            h, w = img.shape[:2]
        nb1 = torch.round(box[:, 0] * w)
        nb2 = torch.round(box[:, 1] * h)
        nb3 = torch.round(box[:, 2] * w)
        nb4 = torch.round(box[:, 3] * h)
        return torch.vstack((nb1, nb2, nb3, nb4)).type(torch.int).T


def normalize_box(img, box):
    """ Normalize any box format w.r.t image size
    Args:
        img: image of type numpy array or torch Tensor
        box: box of format xyxy or cxcywh (haven't tested with xywh)
    Example:
        >>> image = np.zeros((300, 300, 3))
        >>> xyxy = np.asarray([
        >>>     [100, 100, 200, 200]
        >>> ])
        >>> norm_xyxy = normalize_box(image, xyxy)
        >>> print(norm_xyxy)
         [[0.333333 0.333333 0.666667 0.666667]]
    """
    if isinstance(box, (tuple, list)):
        height, width = img.shape[:2]
        return [round(box[0] / width, 6), round(box[1] / height, 6), round(box[2] / width, 6), round(box[3] / height, 6)]
    elif isinstance(box, np.ndarray):
        height, width = img.shape[:2]
        return np.vstack((np.round(box[:, 0] / width, 6), np.round(box[:, 1] / height, 6),
                          np.round(box[:, 2] / width, 6), np.round(box[:, 3] / height, 6))).T
    elif isinstance(box, torch.Tensor):
        if isinstance(img, torch.Tensor):
            width, height = img.shape[2:]  # img shape (batch, c, w, h)
        else:
            height, width = img.shape[:2]
        return torch.vstack((box[:, 0] / width, box[:, 1] / height,
                             box[:, 2] / width, box[:, 3] / height)).T

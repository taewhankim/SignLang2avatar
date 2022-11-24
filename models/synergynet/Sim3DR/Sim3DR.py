# coding: utf-8

from . import _init_paths
import numpy as np
import Sim3DR_Cython
from PIL import Image, ImageDraw


def get_normal(vertices, triangles):
    normal = np.zeros_like(vertices, dtype=np.float32)
    Sim3DR_Cython.get_normal(normal, vertices, triangles, vertices.shape[0], triangles.shape[0])
    return normal


def rasterize(vertices, triangles, colors, face_size,bg=None,
              height=None, width=None, channel=None,
              reverse=False):
    if bg is not None:
        height, width, channel = bg.shape
    else:
        assert height is not None and width is not None and channel is not None
        bg = np.zeros((height, width, channel), dtype=np.float32)
    # height, width = list(map(int,face_size[2:]))
    buffer = np.zeros((height, width), dtype=np.float32) - 1e8
    bg = np.ones((height, width, channel), dtype=np.uint8)*255
    # buffer = np.zeros((int(face_size[3]), int(face_size[2])), dtype=np.float32) - 1e8
    # bg = np.ones((int(face_size[3]), int(face_size[2]), channel), dtype=np.uint8)*255

    if colors.dtype != np.float32:
        colors = colors.astype(np.float32)
    Sim3DR_Cython.rasterize(bg, vertices, triangles, colors, buffer, triangles.shape[0], height, width, channel,
                            reverse=reverse)
    return bg
    # bg_size = bg_size.astype(np.uint8)
    # Sim3DR_Cython.rasterize(bg_size, vertices, triangles, colors, buffer, triangles.shape[0], bg_size.shape[0], bg_size.shape[1], channel,
    #                         reverse=reverse)
    # return bg_size

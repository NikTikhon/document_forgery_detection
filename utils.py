import numpy as np
from skimage.util import view_as_windows
import cv2 as cv
import torch


def fix_bbox(bbox: np.ndarray):
    width_min = bbox[2]
    height_min = bbox[0]
    width_max = bbox[3]
    height_max = bbox[1]
    height_min, height_max, width_min, width_max = resize_image(height_min, height_max, width_min, width_max)
    return height_min, height_max, width_min, width_max


def get_padding(val_1, val_2):
    dif = abs(val_2 - val_1)
    pad_1 = dif // 2
    pad_2 = dif - pad_1
    return pad_1, pad_2


def resize_image(h_min, h_max, w_min, w_max):
    shape_cnn = (64, 64)
    diff_h = h_max - h_min
    diff_w = w_max - w_min
    if diff_h < 64 and diff_w < 64:
        pad_h_top, pad_h_bot = get_padding(diff_h, shape_cnn[0])
        pad_w_left, pad_w_right = get_padding(diff_w, shape_cnn[1])
        return h_min - pad_h_bot, h_max + pad_h_top, w_min - pad_w_left, w_max + pad_w_right
    elif diff_h >= 64 and diff_w < 64:
        pad_w_left, pad_w_right = get_padding(diff_w, shape_cnn[1])
        return h_min, h_max, w_min - pad_w_left, w_max + pad_w_right
    elif diff_h < 64 and diff_w >= 64:
        pad_h_top, pad_h_bot = get_padding(diff_h, shape_cnn[0])
        return h_min - pad_h_bot, h_max + pad_h_bot, w_min, w_max
    return h_min, h_max, w_min, w_max


def drow_mask(image: np.ndarray, bbox: tuple, prediction):
    mask = np.copy(image)
    if prediction:
        color = (0, 0, 255)
        mask = draw_contour(mask, bbox, color)
    else:
        color = (0, 255, 0)
        mask = draw_contour(mask, bbox, color)
    return mask


def get_patches(image_mat, stride):
    window_shape = (64, 64, 3)
    windows = view_as_windows(image_mat, window_shape, step=stride)
    patches = []
    for m in range(windows.shape[0]):
        for n in range(windows.shape[1]):
            patches += [(windows[m][n][0], (m, n))]
    return patches


def bbox_patch(image, patch, stride):
    min_y = patch[1][1] * stride
    max_y = patch[1][1] * stride + 64
    min_x = patch[1][0] * stride
    max_x = patch[1][0] * stride + 64
    color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    image = draw_contour(image, (min_x, max_x, min_y, max_y), color)
    return image


def draw_contour(image: np.ndarray, bbox, color: tuple):
    min_x = bbox[0]
    max_x = bbox[1]
    min_y = bbox[2]
    max_y = bbox[3]
    contours = [np.array([[min_y, min_x], [min_y, max_x], [max_y, max_x], [max_y, min_x]])]
    image = cv.drawContours(image, contours, 0,
                            color, thickness=2)
    return image


def get_prediction(model, tensor):
    with torch.no_grad():
        model.eval()
        return model(tensor)

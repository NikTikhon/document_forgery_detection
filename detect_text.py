import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings
from craft_text_detector import (
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)
warnings.filterwarnings("ignore")


def bbox_contain(bbox_1, bbox_2):
    contain_h_left = (bbox_1[0] - bbox_2[0])
    contain_h_right = (bbox_2[1] - bbox_1[1])

    contain_w_left = (bbox_1[2] - bbox_2[2])
    contain_w_right = (bbox_2[3] - bbox_1[3])
    thr = -7
    if (contain_h_left >= thr and contain_h_right >= thr) and (contain_w_left >= thr and contain_w_right >= thr):
        return 1, 0
    if (contain_h_left < -thr and contain_h_right < -thr) and (contain_w_left < -thr and contain_w_right < -thr):
        return 0, 1
    return 0, 0


def inner_contours(bboxes):
    bboxes_contain = np.zeros(len(bboxes))
    for i in range(len(bboxes) - 1):
        for j in range(i, len(bboxes) - 1):
            bbox1_contain, bbox2_contain = bbox_contain(bboxes[i], bboxes[j + 1])
            if bbox1_contain:
                if bboxes_contain[i] == 0:
                    bboxes_contain[i] = 1
            elif bbox2_contain:
                if bboxes_contain[j + 1] == 0:
                    bboxes_contain[j + 1] = 1
    bboxes_not_contain = []
    for ind in range(len(bboxes)):
        if bboxes_contain[ind] == 0:
            bboxes_not_contain.append(bboxes[ind])
    return bboxes_not_contain


def get_textBBox(image: np.ndarray):
    CUDA = False
    if torch.cuda.is_available():
        CUDA = True
    refine_net = load_refinenet_model(cuda=CUDA)
    craft_net = load_craftnet_model(cuda=CUDA)
    prediction_result = get_prediction(
        image=image,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.8,
        link_threshold=0.1,
        low_text=0.1,
        cuda=False,
        long_size=1280
    )
    regions = prediction_result["boxes"]
    list_bbox = []
    for region in regions:
        region = np.copy(region).astype(int)
        width_heiht_min = np.min(region, axis=0)
        width_heiht_max = np.max(region, axis=0)
        width_min = width_heiht_min[0]
        height_min = width_heiht_min[1]
        width_max = width_heiht_max[0]
        height_max = width_heiht_max[1]
        list_bbox.append((height_min, height_max, width_min, width_max))
    list_bbox = inner_contours(list_bbox)
    empty_cuda_cache()
    return list_bbox


def get_masks(image_path: str):
    image = cv2.imread(image_path)
    masks = get_textBBox(image)
    return masks

import numpy as np
import cv2 as cv

def get_img_filenames(task):
    nir_img_filenames = []
    rgb_img_filenames = []
    for y in range(1,4):
        nir_img_filenames.append("imgs/{}/C0_00000{}.png".format(task,y))
        rgb_img_filenames.append("imgs/{}/C1_00000{}.png".format(task,y))
    return nir_img_filenames, rgb_img_filenames

def get_biggest_connected_component(im_th):
    (numLabels, labels, stats, _) = cv.connectedComponentsWithStats(im_th, 4)
    max_area = -np.inf
    j_mask = -1
    for j in range(1,numLabels):
        w = stats[j, cv.CC_STAT_WIDTH]
        h = stats[j, cv.CC_STAT_HEIGHT]
        area_bbox = h*w
        if (area_bbox>max_area):
            max_area = area_bbox
            j_mask = j
    return (labels == j_mask).astype("uint8") * 255
    
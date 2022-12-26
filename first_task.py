import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import utils
import time 

nir_img_filenames, rgb_img_filenames = utils.get_img_filenames("first_task",start_dir='.')

def task1(nir_img_path, rgb_img_path):
    #t1 = time.time_ns()
    # Read imgs
    nir_img = cv.imread(nir_img_path,0)
    rgb_img = cv.imread(rgb_img_path)

    # Threshold the img
    im_th = cv.adaptiveThreshold(nir_img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,15,3)

    # Invert
    im_th = cv.bitwise_not(im_th)

    # Median filter to remove Salt-and-Pepper like noise
    im_th = cv.medianBlur(im_th,3)

    # Get the binary mask as the biggest connected components in the image (see the report for more explanations)
    binary_mask = utils.get_biggest_connected_component(im_th)

    print(binary_mask.shape)

    h, w = binary_mask.shape[:2]
    op_mask = np.zeros((h+2, w+2), np.uint8)
    cv.floodFill(binary_mask, op_mask, (int(h/2),int(w/2)), 255)

    # Get the contours of the app
    contours, _ = cv.findContours(binary_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    cv.drawContours(rgb_img, contours, -1, (0,255,0), 1)
    
    cv.imshow('{}'.format(rgb_img_path),rgb_img)
    cv.waitKey(2000)
    cv.destroyAllWindows()

    # Apply the mask to the nir img
    nir_fruit = cv.bitwise_and(nir_img,nir_img,mask=binary_mask)

    d = 9
    sigma = 50
    blurred_nir_fruit = cv.bilateralFilter(nir_fruit,d,sigma,sigma)
    edges = cv.Canny(blurred_nir_fruit,60,120)

    # Delete external edge of the fruit and mantain only the defects' edge
    background = 255 - binary_mask
    background_dilated = cv.dilate(background,cv.getStructuringElement(cv.MORPH_ELLIPSE,(23,23)),iterations = 2)
    edges_intern = cv.subtract(edges,background_dilated)

    # Create a binary mask for the 
    structuringElement = cv.getStructuringElement(cv.MORPH_ELLIPSE, (40, 40))
    defects_binary_mask = cv.morphologyEx(edges_intern, cv.MORPH_CLOSE, structuringElement)

    (numLabels, _, _, _) = cv.connectedComponentsWithStats(defects_binary_mask, 4)
    print("Total defects detected : ",numLabels-1)  

    contours, hierarchy = cv.findContours(defects_binary_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(rgb_img, contours, -1, (0,0,255), 1)
    #t2 = time.time_ns()
    #print('time for 1 image: ', (t2-t1)/1000000)
    cv.imshow('{}'.format(rgb_img_path),rgb_img)
    cv.waitKey(2000)
    cv.destroyAllWindows()



# Invoke the function for all images
for i in range(len(nir_img_filenames)):
    task1(nir_img_filenames[i], rgb_img_filenames[i])

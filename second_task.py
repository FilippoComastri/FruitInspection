import cv2 as cv
import numpy as np
import utils

nir_img_filenames, rgb_img_filenames = utils.get_img_filenames("second_task",start_dir='.')
samples_filenames = utils.get_samples_filenames(start_dir='.')

def task2(nir_img_path, rgb_img_path):
    
    nir_img = cv.imread(nir_img_path,0)
    rgb_img = cv.cvtColor(cv.imread(rgb_img_path),cv.COLOR_BGR2RGB)
    samples = []
    for i in range(len(samples_filenames)):
        samples.append(cv.cvtColor(cv.imread(samples_filenames[i]),cv.COLOR_BGR2RGB))

    print('Number of samples ',len(samples))

    # Threshold the img
    im_th = cv.adaptiveThreshold(nir_img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,15,3)

    # Invert
    im_th = cv.bitwise_not(im_th)

    # Median filter to remove Salt-and-Pepper like noise
    im_th = cv.medianBlur(im_th,3)

    # Get the binary mask as the biggest connected components in the image (see the report for more explanations)
    binary_mask = utils.get_biggest_connected_component(im_th)

    # Fill the binary mask
    h, w = binary_mask.shape[:2]
    op_mask = np.zeros((h+2, w+2), np.uint8)
    cv.floodFill(binary_mask, op_mask, (int(h/2),int(w/2)), 255)

    # Convert the samples into LUV colorspace
    luv_samples = []
    for i in range(len(samples)):
        luv_samples.append(cv.cvtColor(samples[i],cv.COLOR_RGB2LUV))
    
    # Convert the image into LUV colorspace
    luv_img = cv.cvtColor(rgb_img,cv.COLOR_RGB2LUV)

    # Prepare the array for the reference color
    reference_color = np.zeros((1,2),dtype=np.float64)
    
    mean_tot = 0
    covar_tot = 0

    # Iterate over the given samples 
    for i in range(len(luv_samples)):
        # Reshape the sample to a 2 column array
        sample_reshaped = luv_samples[i][:,:,1:3].reshape(luv_samples[i].shape[0]*luv_samples[i].shape[1],2)
        # Calculate the covariance matrix of the given sample
        covar, mean = cv.calcCovarMatrix(sample_reshaped,None,flags=cv.COVAR_ROWS | cv.COVAR_NORMAL | cv.COVAR_SCALE)
        mean_tot += mean
        covar_tot += covar

    # The reference color is the mean of the sum of the means of the samples
    reference_color = mean_tot/len(luv_samples)

    # Calcuate the inverse of the covariance matrix with SVD method 
    # because cv2.Mahalanobis function prefers the inverse calculated with this method
    inv_cov = cv.invert(covar_tot, cv.DECOMP_SVD)

    # Prepare the array where to store the binary mask of the russets
    russet = np.zeros((luv_img.shape[0],luv_img.shape[1]),dtype = np.uint8)

    for i in range(luv_img.shape[0]):
        for j in range(luv_img.shape[1]):
            # Only pixels belonging to the apple
            if binary_mask[i][j]==255:
                # Pixel-wise Mahalanobis distance
                mahal_dst = cv.Mahalanobis(np.float64(reference_color.flatten()),np.float64(luv_img[i,j,1:3]),inv_cov[1])
                # If distance is under a certain threshold, the pixel is considered belonging to a russet
                if mahal_dst<1.5:
                    russet[i,j]=255
    
    # Dilate the background to delete edges of the apple and mantain only russets
    background = 255 - binary_mask
    background_dilated = cv.dilate(background,cv.getStructuringElement(cv.MORPH_ELLIPSE,(49,49)),iterations = 1)
    russet_without_borders = cv.subtract(russet,background_dilated)

    # Apply closing to obtain binary masks of russets
    structuringElement = cv.getStructuringElement(cv.MORPH_ELLIPSE, (30, 30))
    final_russet = cv.morphologyEx(russet_without_borders, cv.MORPH_CLOSE, structuringElement)

    # Apply binary mask to rgb image
    russets_rgb = cv.bitwise_and(rgb_img,rgb_img,mask=final_russet)

    cv.imshow('{}'.format(rgb_img_path),cv.cvtColor(russets_rgb,cv.COLOR_RGB2BGR))
    cv.waitKey(2000)
    cv.destroyAllWindows()
    
    (numLabels,_, stats, centroids) = cv.connectedComponentsWithStats(final_russet, 4)
    output = cv.cvtColor(luv_img.copy(),cv.COLOR_LUV2RGB)
    for j in range(1,numLabels):
        w = stats[j, cv.CC_STAT_WIDTH]
        h = stats[j, cv.CC_STAT_HEIGHT]
        (cX, cY) = centroids[j]
        area = stats[j, cv.CC_STAT_AREA]
        if (area>10):
            cv.circle(output, (int(cX), int(cY)), int(w/2), (0, 0, 255), 1)
    
    cv.imshow('{}'.format(rgb_img_path),cv.cvtColor(output,cv.COLOR_RGB2BGR))
    cv.waitKey(2000)
    cv.destroyAllWindows()




    

for i in range(len(nir_img_filenames)):
    task2(nir_img_filenames[i], rgb_img_filenames[i])


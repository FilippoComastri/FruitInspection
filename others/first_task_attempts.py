# USARE OTSU INVECE CHE ADAPTIVE
th, im_th = cv.threshold(nir_img,None,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

plt.imshow(im_th,'gray',vmin=0,vmax=255)
plt.show()

# FLOODFILLING THE BACKGRUND AND INVERT RATHER THAN FLOODFILLING FROM THE CENTER 
im_floodfill = []
op_mask = []
im_floodfill_inv = []
mask = []
for i in range(N):
    # Copy the thresholded image.
    im_floodfill.append(im_th[i].copy())
    
    # Mask used to flood filling.
    # The size needs to be 2 pixels than the image.
    h, w = im_th[i].shape[:2]
    op_mask.append(np.zeros((h+2, w+2), np.uint8))
    
    # Floodfill from point (0, 0)
    cv.floodFill(im_floodfill[i], op_mask[i], (0,0), 255)
    cv.floodFill(im_floodfill[i], op_mask[i], (0,255), 255)
    cv.floodFill(im_floodfill[i], op_mask[i], (255,0), 255)
    cv.floodFill(im_floodfill[i], op_mask[i], (255,0), 255)
    
    # Invert floodfilled image
    im_floodfill_inv.append(cv.bitwise_not(im_floodfill[i]))
    
    # Combine the two images to get the foreground.
    mask.append(cv.bitwise_or(im_th[i],im_floodfill_inv[i]))

plot_images(im_th,title='Thresholded image')
plot_images(im_floodfill_inv,title='Inverted flood-filled image')
plot_images(mask,title='Binary mask of the fruit')

# REFINE THE MASK, MAINTAINING ONLY THE BIGGER CONNECTED COMPONENT
refined_mask = []
for i in range(N):   
    (numLabels, labels, stats, _) = cv.connectedComponentsWithStats(mask[i], 4)
    max = -1000000
    label_max = -1
    for j in range(1,numLabels):
        area = stats[j, cv.CC_STAT_AREA]
        if area>max:
            max = area
            label_max = j
    refined_mask.append((labels == label_max).astype("uint8") * 255)

plot_images(refined_mask,title='refined_mask')
    
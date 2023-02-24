contours = []
output = []

for i in range(N):
    output.append(rgb_img[i].copy())
    contour, _ = cv.findContours(defects_binary_mask[i], cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours.append(contour)
    cv.drawContours(output[i], contours[i], -1, (0,0,255), 2)

plot_images(output,color=True,rgb=False)


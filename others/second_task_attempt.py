# RGB

# sample of the russet
# now taken as a simple pixel. I can optimize calculating a more significant sample
reference_color = np.zeros((1,3),dtype=np.float64)
mean_tot = 0.
covar_tot = 0.
for i in range(len(samples)):
    b,g,r = cv.split(samples[i])
    sample_reshaped = samples[i].reshape(samples[i].shape[0]*samples[i].shape[1],3)
    covar, mean = cv.calcCovarMatrix(sample_reshaped,None,flags=cv.COVAR_ROWS | cv.COVAR_NORMAL | cv.COVAR_SCALE)
    mean_tot += mean
    covar_tot += covar

reference_color = mean_tot/len(samples)

inv_cov = cv.invert(covar_tot, cv.DECOMP_SVD)


for x in range(len(rgb_img)):
    dest_img = np.zeros((rgb_img[x].shape[0],rgb_img[x].shape[1]))
    count = 0
    for i in range(rgb_img[x].shape[0]):
        for j in range(rgb_img[x].shape[1]):
            mahal_dst = cv.Mahalanobis(np.float64(reference_color.flatten()),np.float64(rgb_img[x][i,j,:]),inv_cov[1])
            print(mahal_dst)
            if mahal_dst<2.5:
                dest_img[i,j]=255
                count = count +1
    plt.imshow(rgb_img[x])
    plt.show()

    plt.imshow(dest_img,cmap='gray',vmin=0,vmax=255)
    plt.show()

# HSV

# hsv
hsv_samples = []
for i in range(len(samples)):
    hsv_samples.append(cv.cvtColor(samples[i],cv.COLOR_RGB2HSV))
hsv_img = []
for i in range(len(rgb_img)):
    hsv_img.append(cv.cvtColor(rgb_img[i],cv.COLOR_RGB2HSV))

reference_color = np.zeros((1,2),dtype=np.float64)
mean_tot = 0.
covar_tot = 0.
for i in range(len(hsv_samples)):
    #h,s,v = cv.split(hsv_samples[i])
    sample_reshaped = hsv_samples[i][:,:,:2].reshape(hsv_samples[i].shape[0]*hsv_samples[i].shape[1],2)
    covar, mean = cv.calcCovarMatrix(sample_reshaped,None,flags=cv.COVAR_ROWS | cv.COVAR_NORMAL | cv.COVAR_SCALE)
    mean_tot += mean
    covar_tot += covar

reference_color = mean_tot/len(hsv_samples)
#covar = covar_tot/len(hsv_samples)

inv_cov = cv.invert(covar_tot, cv.DECOMP_SVD)

for x in range(len(hsv_img)):
    dest_img = np.zeros((hsv_img[x].shape[0],hsv_img[x].shape[1]))
    count = 0
    for i in range(hsv_img[x].shape[0]):
        for j in range(hsv_img[x].shape[1]):
            mahal_dst = cv.Mahalanobis(np.float64(reference_color.flatten()),np.float64(hsv_img[x][i,j,:2]),inv_cov[1])
            print(mahal_dst)
            if mahal_dst<0.15:
                dest_img[i,j]=255
                count = count +1
    plt.imshow(cv.cvtColor(hsv_img[x],cv.COLOR_HSV2RGB))
    plt.show()

    plt.imshow(dest_img,cmap='gray',vmin=0,vmax=255)
    plt.show()

# HLS

hls_samples = []
for i in range(len(samples)):
    hls_samples.append(cv.cvtColor(samples[i],cv.COLOR_RGB2HLS))
hls_img = []
for i in range(len(rgb_img)):
    hls_img.append(cv.cvtColor(rgb_img[i],cv.COLOR_RGB2HLS))

reference_color = np.zeros((1,2),dtype=np.float64)
mean_tot = 0.
covar_tot = 0.
for i in range(len(hls_samples)):
    h,l,s = cv.split(hls_samples[i])
    #sample_reshaped = hsv_samples[i][:,:,:2].reshape(hsv_samples[i].shape[0]*hsv_samples[i].shape[1],2)
    h = h.flatten()
    h = h.reshape(h.shape[0],1)
    s = s.flatten()
    s = s.reshape(s.shape[0],1)
    sample_reshaped = np.column_stack((h,s))
    covar, mean = cv.calcCovarMatrix(sample_reshaped,None,flags=cv.COVAR_ROWS | cv.COVAR_NORMAL)
    mean_tot += mean
    covar_tot += covar

reference_color = mean_tot/len(hls_samples)
#covar = covar_tot/len(hsv_samples)

inv_cov = cv.invert(covar_tot, cv.DECOMP_SVD)


for x in range(len(hls_img)):
    dest_img = np.zeros((hls_img[x].shape[0],hls_img[x].shape[1]))
    count = 0
    for i in range(hls_img[x].shape[0]):
        for j in range(hls_img[x].shape[1]):
            h = hls_img[x][i][j][0]
            s = hls_img[x][i][j][2]
            h = h.flatten()
            h = h.reshape(h.shape[0],1)
            s = s.flatten()
            s = s.reshape(s.shape[0],1)
            p = np.column_stack((h,s))
            mahal_dst = cv.Mahalanobis(np.float64(reference_color.flatten()),np.float64(p.flatten()),inv_cov[1])
            #print(mahal_dst)
            if mahal_dst<0.10:
                dest_img[i,j]=255
    plt.imshow(cv.cvtColor(hsv_img[x],cv.COLOR_HSV2RGB))
    plt.show()

    plt.imshow(dest_img,cmap='gray',vmin=0,vmax=255)
    plt.show()

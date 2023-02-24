# prepare the arrays 
reference_color = np.zeros((1,2),dtype=np.float64)
mean_tot = np.zeros((1,2),dtype=np.float64)
covar_tot =  np.zeros((2,2),dtype=np.float64)

for i in range(len(luv_samples)):
    
    # consider only UV part and reshape into a Nx2 matrix
    sample_reshaped = luv_samples[i][:,:,1:3].reshape(luv_samples[i].shape[0]*luv_samples[i].shape[1],2)

    # calculate covariance matrix and mean vector
    covar, mean = cv.calcCovarMatrix(sample_reshaped,None,flags=cv.COVAR_ROWS | cv.COVAR_NORMAL | cv.COVAR_SCALE)
    
    # accumulate
    mean_tot += mean
    covar_tot += covar

reference_color = mean_tot/len(luv_samples)

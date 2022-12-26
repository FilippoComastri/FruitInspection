from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors

r, g, b = cv.split(rgb_img[0])
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")
pixel_colors = rgb_img[0].reshape((np.shape(rgb_img[0])[0]*np.shape(rgb_img[0])[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()
axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Red")
axis.set_ylabel("Green")
axis.set_zlabel("Blue")
plt.show()

hsv_img = cv.cvtColor(rgb_img[0],cv.COLOR_RGB2HSV)
h, s, v = cv.split(hsv_img)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()

hls_img = cv.cvtColor(rgb_img[0],cv.COLOR_RGB2HLS)
h, l, s = cv.split(hls_img)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

axis.scatter(h.flatten(), l.flatten(), s.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Lightness")
axis.set_zlabel("Saturation")
plt.show()

hls_img = cv.cvtColor(rgb_img[0],cv.COLOR_RGB2LUV)
h, l, s = cv.split(hls_img)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

axis.scatter(h.flatten(), l.flatten(), s.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("L")
axis.set_ylabel("U")
axis.set_zlabel("V")
plt.show()
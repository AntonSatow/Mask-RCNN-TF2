import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import openpyxl
from openpyxl import Workbook


original_image = cv.imread("19_frame.jpg")
resize_image = cv.resize(original_image, (1024, 768))

#bilateral_blur = cv.bilateralFilter(resize_image, 9, 75, 75)

#grey_image = cv.cvtColor(resize_image, cv.COLOR_BGR2GRAY)
hist = cv.calcHist([resize_image], [0], None, [256], [0, 256])

spike = np.argmax(hist)
mask = resize_image == spike
resize_image[mask] = [0, 0, 255]

while True:
    #side_by_side = np.hstack((resize_image, bilateral_blur))
    cv.imshow("Original Image vs Gaussian Blur", resize_image)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break


#image = original_image.copy()

#original_image, image = cv.resize(original_image, (640, 480)), cv.resize(image, (640, 480))
#grey_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

#lowpass filter
#kernel = np.array([[1,1,1], [1,1,1], [1,1,1]]) * (1/50.0)
#image = cv.filter2D(image, -1, kernel)

# Create a lookup table
#lut = np.zeros(256, dtype = image.dtype )# create empty array

#for i in range(256):
    #lut[i] = np.clip(pow(i / 255.0, 0.5) * 255.0, 0, 255)

# Apply the lookup table
#image = cv.LUT(image, lut)

#Draw contours to images
#image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#ret, thresh = cv.threshold(image, 127, 255, cv.THRESH_BINARY)

# Find contours
#contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#contour_image = original_image.copy()
#cv.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

#histogram
# create the histogram
#hist,bins = np.histogram(image.flatten(),256,[0,256])
 
#cdf = hist.cumsum()
#cdf_normalized = cdf * float(hist.max()) / cdf.max()
 
#plt.plot(cdf_normalized, color = 'b')
#plt.hist(image.flatten(),256,[0,256], color = 'r')
#plt.xlim([0,256])
#plt.legend(('cdf','histogram'), loc = 'upper left')
#plt.show()
#cdf_m = np.ma.masked_equal(cdf,0)
#cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
#cdf = np.ma.filled(cdf_m,0).astype('uint8')
#cv.equalizeHist(grey_image, image)
# Stack the original and filtered images side by side
#side_by_side = np.hstack((original_image, contour_image))
#side_by_side = np.hstack((original_image, image))
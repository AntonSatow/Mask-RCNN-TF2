import numpy as np
import cv2 as cv

original_image = cv.imread("test.jpg")
image = original_image.copy()

original_image, image = cv.resize(original_image, (640, 480)), cv.resize(image, (640, 480))

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

# Stack the original and filtered images side by side
#side_by_side = np.hstack((original_image, contour_image))
side_by_side = np.hstack((original_image, image))

cv.imshow("test.jpg", side_by_side)
cv.waitKey(30000)
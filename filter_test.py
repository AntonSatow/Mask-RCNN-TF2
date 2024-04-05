import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import openpyxl
from openpyxl import Workbook



def colour_change(image):
    """
    The function takes an image as input and returns a modified image
    The function increases the intensity of the red,green,blue channels by factor
    
    Parameters:
    image (numpy.ndarray): The image to modify
    
    Returns:
    numpy.ndarray: The modified image
    """
    # Split the image into its individual color channels
    r, g, b = cv.split(image)

    # Increase the intensity of the red channel by factor
    r = cv.convertScaleAbs(r, alpha=0.5, beta=0)

    # Increase the intensity of the green channel by factor
    g = cv.convertScaleAbs(g, alpha=5, beta=0)
    
    # Decrease the intensity of the blue channel by factor
    b = cv.convertScaleAbs(b, alpha=10, beta=0)

    # Merge the color channels back together
    modified_image = cv.merge((r, g, b))
    return modified_image
   

def draw_coordinates(event, x, y, flags, param):
    """
    Display the coordinates of the mouse cursor on the image
    Display the RGB values of the pixel at the cursor position
    Display the image with the coordinates and pixel values on the top right corner
    
    Args:
        event (_type_): The type of mouse event (e.g., cv.EVENT_MOUSEMOVE)
        x (int): The current x-coordinate of the mouse cursor
        y (int): The current y-coordinate of the mouse cursor
        flags (int): Additional flags for the mouse event
        param (image): The image to display the coordinates on
    """
    if event == cv.EVENT_MOUSEMOVE:
        # Copy the original image so we don't draw on it permanently
        img = param.copy()
        
        b, g, r = img[y, x]

        # Draw the coordinates on the image
        cv.putText(img, f"X: {x}, Y: {y}, R: {r}, G: {g}, B: {b}",(10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the image
        cv.imshow("Original Image vs Gaussian Blur", img)

original_image = cv.imread("test_img.png")
original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
height = original_image.shape[0]
bottom_half = original_image[height//2:,:]
resize_image = cv.resize(bottom_half, (3000, 920))


image = colour_change(resize_image)

image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
cv.imshow("Original Image vs Gaussian Blur", image)
# Set the mouse callback function to the draw_coordinates function
cv.setMouseCallback("Original Image vs Gaussian Blur", draw_coordinates, param=image)
while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        break



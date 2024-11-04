import numpy as np
import cv2 as cv

def draw_circle(image, center, radius, color, thickness = 1):
    center_x, center_y = center
    for t in range(thickness):
        for angle in range(0, 360):
            x = int(center_x + (radius + t) * np.cos(np.radians(angle)))
            y = int(center_y + (radius + t) * np.sin(np.radians(angle)))
            if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                image[y, x] = color

image = np.zeros((500, 500, 3), dtype="uint8")
draw_circle(image, (250, 250), 50, (100, 50, 100), thickness=10)  

cv.imshow('Circle', image)
cv.waitKey(0)
cv.destroyAllWindows()

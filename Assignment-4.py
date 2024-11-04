import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img_rgb = cv.imread('D:\WORK\Year-3\Image-Processing\mario.jpg')
img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)
template1 = cv.imread('D:\WORK\Year-3\Image-Processing\mario-coin.jpg',cv.IMREAD_GRAYSCALE)

w,h = template1.shape[::-1]

res = cv.matchTemplate(img_gray,template1,cv.TM_CCOEFF_NORMED)

threshold = 0.8
loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
    cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255),2)

cv.imshow('',img_rgb)
cv.waitKey()



import cv2 as cv

# Asingement1-1
img = cv.imread('Image-Processing\Test-image.jpg')

red , green , blue = cv.split(img)
size1 = (300,300)
size2 = (250,250)
img_resize = cv.resize(img , size1)

RGBimg = cv.vconcat([red ,green ,blue])
img_total = cv.merge([red ,green ,blue])

imgresize_red = cv.resize( red , size2)
imgresize_green = cv.resize( green , size2)
imgresize_blue = cv.resize( blue , size2)

img_total1 = cv.resize(img_total,size1)

imgvertical = cv.vconcat([imgresize_red,imgresize_green,imgresize_blue])

# cv.imshow('Ex1-imshow', imgresize)
# cv.imshow('RGB',imgvertical)
# cv.imshow('MergeIMG',img_total1)

# Asingement1-2

brightnes = cv.convertScaleAbs(img,0,1.5)
bright_resize = cv.resize(brightnes,size1)

totalimage = cv.vconcat([img_resize,bright_resize])
# cv.imshow('brightnes',bright_size)

cv.imshow('plane + brightnes',totalimage)

cv.imwrite('D:\WORK\Year-3\Image-Processing\image_red.jpg', red)
cv.imwrite('D:\WORK\Year-3\Image-Processing\image_green.jpg', green)
cv.imwrite('D:\WORK\Year-3\Image-Processing\image_blue.jpg', blue)
cv.imwrite('D:\WORK\Year-3\Image-Processing\RGB.jpg',imgvertical)
cv.imwrite('D:\WORK\Year-3\Image-Processing\mergeimg.jpg',img_total)
cv.imwrite('D:\WORK\Year-3\Image-Processing\Brightnes.jpg',bright_resize)
cv.imwrite('D:\WORK\Year-3\Image-Processing\plane_brightnes.jpg',totalimage)   

cv.waitKey(0)
cv.destroyAllWindows()
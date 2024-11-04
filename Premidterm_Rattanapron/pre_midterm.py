import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and convert the image to gray-scale
img = cv2.imread('D:/WORK/Year-3/Image-Processing/Premidterm/mergeimg.jpg')  
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('D:/WORK/Year-3/Image-Processing/Premidterm/grayimg.png', gray_image)

# Compute magnitude spectrum
f = np.fft.fft2(gray_image)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1) 
magnitude_spectrum = np.uint8(255 * magnitude_spectrum / np.max(magnitude_spectrum)) 
cv2.imwrite('D:/WORK/Year-3/Image-Processing/Premidterm/magnitude.png', magnitude_spectrum)

# Ideal Lowpass and Highpass Filter masks
rows, cols = gray_image.shape
crow, ccol = rows // 2, cols // 2  

radius = 15
lowpass_mask = np.zeros((rows, cols), np.uint8)
cv2.circle(lowpass_mask, (ccol, crow), radius, 1, thickness=-1)

highpass_mask = np.ones((rows, cols), np.uint8)
cv2.circle(highpass_mask, (ccol, crow), radius, 0, thickness=-1)

# Apply filters
fshift_lowpass = fshift * lowpass_mask
fshift_highpass = fshift * highpass_mask

lowpass_image = np.fft.ifft2(np.fft.ifftshift(fshift_lowpass))
lowpass_image = np.abs(lowpass_image)
lowpass_image = np.uint8(255 * lowpass_image / np.max(lowpass_image)) 

highpass_image = np.fft.ifft2(np.fft.ifftshift(fshift_highpass))
highpass_image = np.abs(highpass_image)
highpass_image = np.uint8(255 * highpass_image / np.max(highpass_image))

# magnitude spectrum
magnitude_lowpass = magnitude_spectrum * lowpass_mask
magnitude_highpass = magnitude_spectrum * highpass_mask

cv2.imwrite('D:/WORK/Year-3/Image-Processing/Premidterm/magnitude_low.png', magnitude_lowpass)
cv2.imwrite('D:/WORK/Year-3/Image-Processing/Premidterm/magnitude_high.png', magnitude_highpass)

# lowpass Highpass
cv2.imwrite('D:/WORK/Year-3/Image-Processing/Premidterm/inverse_low.png', lowpass_image)
cv2.imwrite('D:/WORK/Year-3/Image-Processing/Premidterm/inverse_high.png', highpass_image)

# Combined image
combined_image = cv2.add(lowpass_image, highpass_image)
cv2.imwrite('D:/WORK/Year-3/Image-Processing/Premidterm/inverse.png', combined_image)

# Display  result
plt.subplot(2, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Gray-scale Image')
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(combined_image, cmap='gray')
plt.title('Combined Image')
plt.axis('off')


plt.show()

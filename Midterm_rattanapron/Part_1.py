import cv2
import numpy as np
import matplotlib.pyplot as plt

# โหลดภาพ
image = cv2.imread('D:\WORK\Year-3\Image-Processing\Midterm_rattanapron\part1img.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

# สร้าง histrogram ก่อน
plt.hist(image.ravel(), bins=256, color='gray')
plt.title('Histogram Before')
plt.xlabel('Intensity Value')
plt.ylabel('Pixel Count')
plt.savefig('D:\WORK\Year-3\Image-Processing\Midterm_rattanapron\Result1-Hist-Before.png')
plt.close()

# ลด Contrast และเพิ่ม Brightness
alpha = 1.5  # ค่าContrast
beta = 50    # ค่าBrightness


adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# สร้าง histrogram หลัง
plt.hist(adjusted_image.ravel(), bins=256, color='gray')
plt.title('Histogram After')
plt.xlabel('Intensity Value')
plt.ylabel('Pixel Count')
# บันทึกภาพ Result1-Hist-After
plt.savefig('D:\WORK\Year-3\Image-Processing\Midterm_rattanapron\Result1-Hist-After.png')
plt.close()

# บันทึกภาพ
cv2.imwrite('D:\WORK\Year-3\Image-Processing\Midterm_rattanapron\Result1.png', cv2.cvtColor(adjusted_image, cv2.COLOR_RGB2BGR))  # กลับไปเป็น BGR เพื่อบันทึก

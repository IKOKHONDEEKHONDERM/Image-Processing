import cv2
import numpy as np

# อ่านภาพเข้าไป
image = cv2.imread('D:\WORK\Year-3\Image-Processing\Midterm_rattanapron\part3img.jpg')

# แปลงภาพจาก BGR ไปเป็น HSV เพื่อให้การกรองสีง่ายขึ้น
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# กำหนดช่วงสีสำหรับสีขาว
lower_white = np.array([0, 0, 200])  # ค่าสีขาวต่ำสุด
upper_white = np.array([180, 50, 255])  # ค่าสีขาวสูงสุด

# กรองเฉพาะสีขาวออกมา
mask_white = cv2.inRange(hsv, lower_white, upper_white)
edges = cv2.Canny(mask_white, 100, 200)
contours, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# สร้างภาพใหม่ที่เป็นสีดำทั้งหมด
output = np.zeros_like(image)

# แทนที่บริเวณที่เป็นสีขาวในหน้ากากให้เป็นสีขาว (255, 255, 255)
output[mask_white > 0] = [255, 255, 255]

# บันทึกผลลัพธ์เป็นไฟล์ชื่อ Result3.png
cv2.imwrite('D:\WORK\Year-3\Image-Processing\Midterm_rattanapron\Result3.png', output)
cv2.waitKey(0)
cv2.destroyAllWindows()

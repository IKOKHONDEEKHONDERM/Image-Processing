import cv2
import numpy as np

image = cv2.imread('D:\WORK\Year-3\Image-Processing\Midterm_rattanapron\part2img.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ใช้ Gaussian blur เพื่อลด noise 
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# ใช้แคนนี่
edges = cv2.Canny(blurred, 50, 150)
# หา contours 
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros(gray.shape, dtype="uint8")

# วนลูปผ่าน contours
for contour in contours:
    # หา bounding box ของ contour
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.int32)
    
    # คำนวณอัตราส่วนความกว้างต่อความสูงของ contour
    width = rect[1][0]
    height = rect[1][1]
    
    # ตรวจสอบว่า contour มีอัตราส่วนความกว้างต่อความสูงตามปกติของบาร์โค้ดหรือไม่
    if width > 0 and height > 0:
        aspect_ratio = width / height if width > height else height / width
        if 3.0 < aspect_ratio < 10.0:  # บาร์โค้ดมักมีอัตราส่วนความกว้างต่อความสูงสูง
            # วาด contour ลงบนหน้ากากเป็นสีขาว (255)
            cv2.drawContours(mask, [box], -1, (255), -1)

# พลิกหน้ากากเพื่อลดพื้นที่พื้นหลังให้เป็นสีดำ (0) และพื้นที่บาร์โค้ดให้เป็นสีขาว (255)
barcode_result = cv2.bitwise_and(gray, gray, mask=mask)

cv2.imwrite('D:\WORK\Year-3\Image-Processing\Midterm_rattanapron\Result2-1.png', mask)
cv2.imwrite('D:\WORK\Year-3\Image-Processing\Midterm_rattanapron\Result2-2.png', barcode_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

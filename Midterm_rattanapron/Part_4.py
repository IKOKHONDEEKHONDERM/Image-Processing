import cv2

# อ่านภาพเข้าไป
image = cv2.imread('D:\WORK\Year-3\Image-Processing\Midterm_rattanapron\part4img.png')  # เปลี่ยนเส้นทางของภาพที่ต้องการประมวลผล

# ใช้ Median Filter เพื่อลด noise
denoised_image = cv2.medianBlur(image, 5)  # ใช้ kernel ขนาด 5

# บันทึกผลลัพธ์เป็นไฟล์ชื่อ Result4.png
cv2.imwrite('D:\WORK\Year-3\Image-Processing\Midterm_rattanapron\Result4.png', denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

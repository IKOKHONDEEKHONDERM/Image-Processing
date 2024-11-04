import cv2
import numpy as np
from matplotlib import pyplot as plt

def apply_gamma_correction(image, gamma):
    # คำนวณค่า lookup table สำหรับการปรับ gamma
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ไม่สามารถเปิด webcam ได้")
else:
    print("กด 's' เพื่อถ่ายรูป และ 'q' เพื่อออก")

    while True:
        ret, frame = cap.read()
        
        cv2.imshow('Webcam', frame)

        key = cv2.waitKey(1)

        if key == ord('s'):
            image_path = 'captured_image.jpg'
            cv2.imwrite(image_path, frame)
            print("ถ่ายรูปเรียบร้อยแล้ว!")

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            hist = cv2.calcHist([image], [0], None, [256], [0, 256])

            mean_intensity = np.mean(image)

            print(f"ค่าเฉลี่ยความสว่างของภาพ: {mean_intensity}")

            # ตรวจสอบความสว่างและทำ gamma correction
            if mean_intensity < 100:
                print("ภาพมืดเกินไป ทำการปรับ gamma correction ด้วย y < 1")
                gamma_corrected = apply_gamma_correction(image, 0.5)  # gamma < 1
            elif mean_intensity > 150:
                print("ภาพสว่างเกินไป ทำการปรับ gamma correction ด้วย y > 1")
                gamma_corrected = apply_gamma_correction(image, 1.5)  # gamma > 1
            else:
                print("ความสว่างของภาพเหมาะสมแล้ว")
                gamma_corrected = image

            cv2.imshow('Gamma Corrected Image', gamma_corrected)
            cv2.imwrite('gamma_corrected_image.jpg', gamma_corrected)

            hist_corrected = cv2.calcHist([gamma_corrected], [0], None, [256], [0, 256])
            plt.figure()
            plt.title("Grayscale Histogram")
            plt.xlabel("Bins")
            plt.ylabel("# of Pixels")
            plt.plot(hist_corrected)
            plt.xlim([0, 256])
            plt.show()

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
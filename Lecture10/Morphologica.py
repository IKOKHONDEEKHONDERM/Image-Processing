import cv2
import numpy as np

image_path = 'Lecture10\Image.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

num_objects = len(contours)
print(f"Number of objects: {num_objects}")

# Find areas of all contours
areas = [cv2.contourArea(cnt) for cnt in contours]

# Sort contours by area and keep the two largest
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
largest_contours = sorted_contours[:2]

output_image = np.zeros_like(image)

cv2.drawContours(output_image, largest_contours, -1, (255), thickness=cv2.FILLED)

# Save the result image
cv2.imwrite('D:\WORK\Year-3\Image-Processing\Lecture10\largest_two_objects.png', output_image)

# Find widths of all contours
bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
widths = [w for (x, y, w, h) in bounding_boxes]

min_width = min(widths)
min_width_index = widths.index(min_width)
min_width_contour = contours[min_width_index]

max_width = max(widths)
max_width_index = widths.index(max_width)
max_width_contour = contours[max_width_index]

print(f"Smallest width object index: {min_width_index}, width: {min_width}")
print(f"Largest width object index: {max_width_index}, width: {max_width}")

output_image = image.copy()
cv2.drawContours(output_image, [min_width_contour], -1, (127), 3)
cv2.drawContours(output_image, [max_width_contour], -1, (255), 3)

cv2.imwrite('D:\WORK\Year-3\Image-Processing\Lecture10\smallest_largest_width_objects.png', output_image)

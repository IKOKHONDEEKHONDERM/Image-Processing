import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

def load_image(image_path):
    # Load image in BGR format
    image = cv.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read.")
    return image

def calculate_histogram(image):
    # Convert to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Calculate histogram
    hist = cv.calcHist([gray_image], [0], None, [256], [0, 256])
    return hist

def analyze_brightness(hist):
    # Analyze the histogram to infer brightness
    # total_pixels = np.sum(hist)
    
    # Calculate the cumulative distribution from the histogram
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]
    
    # Determine brightness thresholds (for example, 10% and 90%)
    low_threshold = 0.9
    high_threshold = 0.1
    
    is_dark = cdf_normalized[50] < low_threshold  # Example threshold
    is_bright = cdf_normalized[200] > high_threshold  # Example threshold
    
    return is_dark, is_bright

def adjust_gamma(image, gamma):
    # Apply gamma correction
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    
    return cv.LUT(image, table)

def plot_histogram(hist, title="Histogram"):
    # Plot histogram
    plt.figure()
    plt.title(title)
    plt.xlabel("Intensity Value")
    plt.ylabel("Frequency")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()

def main(image_path):
    # Load the image
    image = load_image(image_path)
    
    # Calculate histogram
    hist = calculate_histogram(image)
    
    # Plot original histogram
    plot_histogram(hist, title="Original Histogram")
    
    # Analyze brightness
    is_dark, is_bright = analyze_brightness(hist)
    
    # Adjust gamma based on brightness analysis
    if is_dark:
        gamma = 0.7  # Example gamma value to brighten the image
    elif is_bright:
        gamma = 1.5  # Example gamma value to darken the image
    else:
        gamma = 1.0  # No change
    
    adjusted_image = adjust_gamma(image, gamma)
    
    # Calculate and plot adjusted histogram
    adjusted_hist = calculate_histogram(adjusted_image)
    plot_histogram(adjusted_hist, title="Adjusted Histogram")
    
    # Display images
    cv.imshow("Original Image", image)
    cv.imshow("Adjusted Image", adjusted_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

image_path = 'D:\WORK\Year-3\Image-Processing\demo.png'
main(image_path)
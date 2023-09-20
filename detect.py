import numpy as np
import cv2
import imutils
import pytesseract
import pandas as pd
import time
import matplotlib.pyplot as plt  # Import matplotlib for image display

image_path = 'images/7.jpg'

img = cv2.imread(image_path, cv2.IMREAD_COLOR)
img = imutils.resize(img, width=500)

# Convert BGR image to RGB for matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the image using matplotlib
plt.imshow(img_rgb)
plt.axis('off')  # Turn off axis labels
plt.show()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 170, 200)

# Find contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

min_area = 1000  # Adjust minimum contour area as needed
max_area = 50000  # Adjust maximum contour area as needed
NumberPlateCnt = None

# Iterate through contours and filter based on area
for contour in contours:
    area = cv2.contourArea(contour)
    if min_area < area < max_area:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            NumberPlateCnt = approx
            break

# Initialize mask with zeros
mask = np.zeros_like(gray, dtype=np.uint8)

# Draw the number plate region on the mask
if NumberPlateCnt is not None:
    cv2.drawContours(mask, [NumberPlateCnt], -1, (255), -1)

    # Apply the mask to the grayscale image
    masked_image = cv2.bitwise_and(gray, gray, mask=mask)

    # Threshold the masked image to enhance text visibility
    _, thresh = cv2.threshold(masked_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Configuration for Tesseract
    config = ('-l eng --oem 1 --psm 6')

    # Run Tesseract OCR on the thresholded image
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    text = pytesseract.image_to_string(thresh, config=config)

    # Data is stored in CSV file
    raw_data = {'date': [time.asctime(time.localtime(time.time()))],
                'v_number': [text]}

    df = pd.DataFrame(raw_data, columns=['date', 'v_number'])
    df.to_csv('data.csv')

    # Print recognized text
    print("Recognized Text:", text)
else:
    print("Number plate contour not found.")

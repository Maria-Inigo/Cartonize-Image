import numpy as np
import cv2

KERNEL_SIZE_BLUR = (25, 25)
KERNEL_SIZE_EDGE = 9

# 1. Read image
image = cv2.imread('images/photographer.jpeg')

# 2. Convert the image to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Filter image with LPF to remove noise (blur/smooth image)
# apply an "average" blur to the image using the current kernel
image_gray_blurred = cv2.blur(image_gray, KERNEL_SIZE_BLUR)
# image_gray_blurred = cv2.GaussianBlur(image_gray,KERNEL_SIZE_BLUR,0)


# 4. Sobel (HPF) Edge Detection
# Combined X and Y Sobel Edge Detection
grad_x = cv2.Sobel(image_gray_blurred, cv2.CV_64F, 1, 0, ksize=KERNEL_SIZE_EDGE)
grad_y = cv2.Sobel(image_gray_blurred, cv2.CV_64F, 0, 1, ksize=KERNEL_SIZE_EDGE)
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
# laplacian = cv2.Laplacian(image_gray_blurred,cv2.CV_64F) # Works better if the objects are well defined, can't edit the kernel
# Convert to binary
ret,binary_image = cv2.threshold(grad,127,255,cv2.THRESH_BINARY)
# Convert to "color"
colored_sobelxy = cv2.cvtColor(binary_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

# Check that the mask is binary
# unique, counts = np.unique(thresh1, return_counts=True)
# print(dict(zip(unique, counts)))

# 5. Mask original image
masked = np.minimum(image, colored_sobelxy)

cv2.imshow('Original', image)
cv2.imshow('Cartoon', masked)
cv2.waitKey(0)
from PIL import Image
import numpy as np
import cv2

# 1. Read image
img = Image.open('IMG_0041.jpg')

# 2. Create Mask 
## Convert the image to grayscale
img_gray = img.convert('L')
## Filter image with LPF to remove noise (blur/smooth image)


# 3. Mask original image
# ASK: Why do I need two frames for AND bit operator?
# masked_image = cv2.bitwise_and(np_image, np_image, mask = np_img_gray)
np_image = np.array(img)
np_img_gray = np.array(img)
masked_image = cv2.bitwise_and(np_image, np_img_gray)

img_gray.show()

Step 1 : Converting a Digital Image to Gray Scale Image

import cv2
import numpy as np
from matplotlib import pyplot as plt
# Conversion of image to gray scale
img = cv2.imread("img1.jpg",0)
cv2.imshow("Grey image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

*******************************************************
Step2 : Apply Canny algorithm foredge dectection on gray scale image.

import cv2
import numpy as np
from matplotlib import pyplot as plt
# Conversion of image to gray scale
img = cv2.imread("img1.jpg",0)
#use of canny algorithm function for edge detection
canny = cv2.Canny(img,100,200)

titles = ("image","canny")
images = (img , canny)
for i in range(2):
    plt.subplot(1 ,2,i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
#cv2.imshow("Grey image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

******************************************************
Step3 : Apply Mser Algorithm for Text region detection in an image

# Import packages
import cv2
import numpy as np
#Create MSER object
mser = cv2.MSER_create()

#Your image path i-e receipt path
img = cv2.imread('img1.jpg')



#Convert to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

vis = img.copy()

#detect regions in gray scale image
regions, _ = mser.detectRegions(gray)

hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

cv2.polylines(vis, hulls, 1, (0, 255, 0))

cv2.imshow('img', vis)

cv2.waitKey(0)

mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

for contour in hulls:

    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

#this is used to find only text regions, remaining are ignored
text_only = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("text only", text_only)

cv2.waitKey(0)
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
cv2.waitKey(0)

#find contours
ctrs, hier = cv2.findContours ( thresh.copy(), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = img[y:y+h, x:x+w]

    # show ROI
    #cv2.imwrite('roi_imgs.png', roi)
    cv2.imshow('charachter'+str(i), roi)
    cv2.rectangle(img,(x,y),( x + w, y + h ),(90,0,255),2)
    cv2.waitKey(0)

cv2.imshow('marked areas',img)
cv2.waitKey(0)

********************************************************
Step4 :  Apply an OCR(Tesseract) for text to be in readable form

import pytesseract as ptrt
ptrt.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import Image

img_unprocessed=Image.open("")
str_frm_img=ptrt.image_to_string(img_unprocessed)

print(str_frm_img)

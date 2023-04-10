import numpy as np
import cv2
import  imutils
import sys
import pytesseract
import pandas as pd
import time

image = cv2.imread('a.jpg')

image = imutils.resize(image, width=500)       #resize lai anh
cv2.imshow("Original Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #Chuyen sang mau gray
# cv2.imshow("1 - Grayscale Conversion", gray)

gray = cv2.bilateralFilter(gray, 11, 17, 17)    #Chuyen sang mau nhi phan gray
# cv2.imshow("2 - Bilateral Filter", gray)

edged = cv2.Canny(gray, 170, 200)   #Bat edge cua hinh anh
cv2.imshow("4 - Canny Edges", edged)
(cnts,_) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# print(cnts)
cnts_new=sorted(cnts, key = cv2.contourArea, reverse = True)[:30] 
# print(cnts_new)
NumberPlateCnt = None
count = 0
for c in cnts_new:
        # print(count)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 00.02 * peri, True)
        if len(approx) == 4:  
            NumberPlateCnt = approx 
            # print(NumberPlateCnt)
            break

mask = np.zeros(gray.shape,dtype = np.uint8)

new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1)
new_image = cv2.bitwise_and(image,image,mask=mask)
# cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)
cv2.imshow("Final_image2",new_image)

config = ('-l eng --oem 1 --psm 3')
# Run tesseract OCR on image
text = pytesseract.image_to_string(new_image, config=config)

#Data is stored in CSV file
raw_data = {'date': [time.asctime( time.localtime(time.time()) )], 
        'v_number': [text]}

df = pd.DataFrame(raw_data, columns = ['date', 'v_number'])
df.to_csv('data.csv')

# Print recognized text
print(text)

cv2.waitKey(0)
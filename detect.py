import cv2
import os
import numpy as np

#Changing my directory to the project folder
os.chdir( 'data/' )

#Read input image
img = cv2.imread('1.jpg')
os.chdir(r'C:\Users\elcot\Desktop\FTC\target')

#Grey scale image
img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

#Noise removal
noise_removal = cv2.bilateralFilter(img_gray,9,75,75)
#ret,thresh_image = cv2.threshold(noise_removal,0,255,cv2.THRESH_OTSU)

#Find 
canny_image = cv2.Canny(noise_removal,60,3*60)
#canny_image = cv2.Canny(thresh_image,250,255)

#Absolute values
canny_image = cv2.convertScaleAbs(canny_image)

#Convoluting the image
kernel = np.ones((3,3), np.uint8)

dilated_image = cv2.dilate(canny_image,kernel,iterations=1)

#Finding edges
img1,contours, h = cv2.findContours(dilated_image, 1, 2)

#Sort the contours
contours= sorted(contours, key = cv2.contourArea, reverse = True)[:1]

pt = (180, 3 * img.shape[0] // 4)

#Drawing the border
for cnt in contours:
    
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    print(len(approx))
    if len(approx) ==6 :
        cv2.drawContours(img,[cnt],-1,(0,0,255),7)        
    elif len(approx) == 7:
        cv2.drawContours(img,[cnt],-1,(0,0,255),7)
    elif len(approx) == 8:
        cv2.drawContours(img,[cnt],-1,(0,0,255),7)
    elif len(approx) > 8:
        cv2.drawContours(img,[cnt],-1,(0,0,255),7)

#Extracting more features
corners    = cv2.goodFeaturesToTrack(img_gray,20,0.06,25)
#corners    = cv2.goodFeaturesToTrack(img_gray,20,0.06,25)
corners    = np.float32(corners)

for    item in    corners:
    x,y    = item[0]
    cv2.circle(img,(x,y),10,(0,0,0),-1)
#Save image
cv2.imwrite("1.jpg",img)




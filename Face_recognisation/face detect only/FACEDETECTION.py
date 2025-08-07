#importing open cv library
import cv2

#dataset
traindData=cv2.CascadeClassifier('Face.xml')

#choose a image
img=cv2.imread('one.jpg')

#Coversion to black and white (grayscale)
grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detect face
facecoordinate=traindData.detectMultiScale(grayimg)
#coordinate of grayimg-->[[558 331 235 235]]

x,y,w,h=facecoordinate[0]
cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

#display image
cv2.imshow('Single Person',img)

#pause the execution of the program until any key is pressed 
cv2.waitKey()


print ('end of program')
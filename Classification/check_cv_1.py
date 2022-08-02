import cv2

input=cv2.imread('E:\pythonProject\Classification\sachin.jpg') # to load image
# print(input.shape)
# cv2.imshow('Hello',input) # first paramter-title, second parameter--image variable
# cv2.waitKey(0)
gray=cv2.cvtColor(input,cv2.COLOR_BGR2GRAY) #convert rgb pic to gray color pic
# cv2.imshow('gray',gray)
# cv2.waitKey()
#

face_detector=cv2.CascadeClassifier('E:\pythonProject\Classification\haarcascade\haarcascade_frontalface_default.xml')
eye_detector=cv2.CascadeClassifier('E:\pythonProject\Classification\haarcascade\haarcascade_eye.xml')
eyes=eye_detector.detectMultiScale(gray,1.1,3) # this method accepts only gray images
faces=face_detector.detectMultiScale(gray,1.1,3)

for (x,y,w,h) in faces:
    cv2.rectangle(input,(x,y),(x+w,y+h),(0,255,0),2)
for (x,y,w,h) in eyes:
    cv2.rectangle(input,(x,y),(x+w,y+h),(0,0,255),2)

cv2.imshow('img',input)
cv2.waitKey()


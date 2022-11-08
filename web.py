import cv2
# reading the webcam
cap=cv2.VideoCapture(0) # a method that signifies the starting of the webcam 
imgcont=0 #let's assume the number of images gotten is 0
while True :
    ret , img= cap.read() # reading the webcam
    cv2.imshow('webcam',img)  #showing the webcam to the user 
    k=cv2.waitKey(1) #to keep the video playing nonstop
    if k==27: # if the escape key is been pressed, the app will stop 
     break
   # if the spacebar key is been pressed screenshots will be taken
    elif k==32:
         # the format for storing the images scrreenshotted
        imgname="opencv_frame{}.png".format(imgcont) 
        # saves the image as a png file
        cv2.imwrite(imgname,img)
        print("screenshot taken")
         # the number of images automaticallly increases by 1
        imgcont+=1


cap.release()
cv2.destroyAllWindows()

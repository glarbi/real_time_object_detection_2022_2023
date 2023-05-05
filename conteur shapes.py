import cv2
import csv
import os

input_path = "/"
with open(".csv","w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
  
        dir_list = os.listdir(input_path)

        for im in dir_list:
            
            image = cv2.imread(input_path+im)
          
            original = image.copy()
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
            dilate = cv2.dilate(thresh, kernel, iterations=1)

            cnts,hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image,cnts,-1,(255,255,0),3)
            
            cnt=cnts[0]
          
            
            points = cnt.tolist()
            csvWriter.writerow(points)

        
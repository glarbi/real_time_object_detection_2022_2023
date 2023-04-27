import cv2
import os

# Get the list of all files and directories

input_path = "C:/Users/Home/Documents/Univ_Batna/Master II - IAM/Projets/Projets_2022_2023/Real-Time Plastic Waste Detection/Application/detection_waste/test/plastic/"
output_path = "C:/Users/Home/Documents/Univ_Batna/Master II - IAM/Projets/Projets_2022_2023/Real-Time Plastic Waste Detection/Application/detection_waste/out/test/plastic/"
dir_list = os.listdir(input_path)

for im in dir_list:
    # Load image, grayscale, Gaussian blur, Otsu's threshold, dilate
    image = cv2.imread(input_path+im)
    original = image.copy()
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours, obtain bounding box coordinates, and extract ROI
    savedContour = -1
    maxArea = 0.0
    cnts,hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ##########################################
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    image_number = 0
    for i in range(1, len(cnts)):
        area = cv2.contourArea(cnts[i])
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnts[i])
            #cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
            ROI = original[y:y+h, x:x+w]
            outfname = "{0}ROI_{1}_{2}.png".format(output_path, im, image_number)
            cv2.imwrite(outfname, ROI)
            image_number += 1
    ##########################################
#     for i in range(0, len(cnts)):
#         area = cv2.contourArea(cnts[i])
#         if area > maxArea:
#             maxArea = area
#             savedContour = i
# #    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#     image_number = 0
#
#     x,y,w,h = cv2.boundingRect(cnts[savedContour])
#     cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
#     ROI = original[y:y+h, x:x+w]
#     outfname = "{0}ROI_{1}_{2}.png".format(output_path, im, image_number)
#     cv2.imwrite(outfname, ROI)
#     image_number += 1

    # cv2.imshow('image', image)
    # cv2.imshow('thresh', thresh)
    # cv2.imshow('dilate', dilate)
    # cv2.waitKey()
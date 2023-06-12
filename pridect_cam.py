import math
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from PIL import Image
from keras.preprocessing import image
from keras.utils import load_img, img_to_array
from shapely import Polygon 
import os

def getROIs(image):
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours, obtain bounding box coordinates, and   ROI

    cnts, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)


    ret = pd.DataFrame(columns = ['x', 'y', 'w', 'h'])
    ROIs = []
    gray_images = []
    contours = []
    for i in range(0, len(cnts)):
        gray_image = np.zeros((1000, 1000), dtype='uint8')
        h = len(cnts[i])
        d = len(cnts[i][0])
        w = len(cnts[i][0][0])
        cnt = np.resize(cnts[i], (h, w))
        poly = Polygon(cnt.tolist())

        area = poly.area
        #area = cv2.contourArea(cnts[i])
        if not (math.isclose(poly.minimum_rotated_rectangle.area, poly.area, rel_tol=0.05)):  # If the contour isn't rectangle tolerance = 5%
            if area >5000:
                cv2.drawContours(gray_image, [cnts[i]], -1, (255, 255, 255), 3)
                gray_images.append(gray_image)

                x, y, w, h = cv2.boundingRect(cnts[i])
                ROIs.append(original[y:y + h, x:x + w])
                #ROIs.append(gray[y:y + h, x:x + w])
                ret = pd.concat([ret, pd.DataFrame([[x, y, w, h]], columns=['x', 'y', 'w', 'h'])], ignore_index=True)
                contours.append(cnts[i])

    return ROIs, gray_images, contours, ret


model = load_model('detection_waste_plastic.h5')

###################### Predict from webcam ########################################"


#test_datagen = image.ImageDataGenerator(rescale=1./255)
test_datagen = image.ImageDataGenerator(rescale=1.0 / 255.0)
folder_path = 'tst/'
test_datagen = image.ImageDataGenerator(rescale=1.0 / 255.0)
# Iterate over the images in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Only process image files
        # Read the image
        image_path = os.path.join(folder_path, filename)
        frame = cv2.imread(image_path)

        # Process the image
        myROIs, my_gray_images, my_contours, coordinates = getROIs(frame)
        if len(my_gray_images) > 0:
            i = 0
            while i < len(my_gray_images):
            #im = Image.fromarray(myROIs[i], 'RGB')
                im = Image.fromarray(my_gray_images[i], 'L')

                im = im.resize((250, 250))
                im1 = img_to_array(im)
                im1 = cv2.cvtColor(im1, cv2.COLOR_GRAY2RGB)
                img = test_datagen.standardize(np.copy(im1))
                img_array = np.expand_dims(np.asarray(img), axis=0)
                prediction = model.predict(img_array)
                if prediction[0] >0.3:
                    label = "plastic " + str(prediction[0])
                    df = coordinates.iloc[i]
                    x = df['x']
                    y = df['y']
                    w = df['w']
                    h = df['h']
                    cv2.putText(frame, label, (x, y+10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 0,255), 1)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 2)
                    #cv2.drawContours(frame, [my_contours[i]], -1, (255, 255, 255), 3)
                i += 1

        output_path = os.path.join(folder_path, 'result_' + filename)
        cv2.imwrite(output_path, frame)
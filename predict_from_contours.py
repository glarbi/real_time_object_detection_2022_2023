import math

import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from PIL import Image
from keras.preprocessing import image
from keras.utils import load_img, img_to_array
from shapely import Polygon

def getROIs(image):
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours, obtain bounding box coordinates, and extract ROI

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
            if area > 3000:
                cv2.drawContours(gray_image, [cnts[i]], -1, (255, 255, 255), 3)
                gray_images.append(gray_image)

                x, y, w, h = cv2.boundingRect(cnts[i])
                ROIs.append(original[y:y + h, x:x + w])
                #ROIs.append(gray[y:y + h, x:x + w])
                ret = pd.concat([ret, pd.DataFrame([[x, y, w, h]], columns=['x', 'y', 'w', 'h'])], ignore_index=True)
                contours.append(cnts[i])

    return ROIs, gray_images, contours, ret


model = load_model('shape_based_detection_waste_plastic_100epochs.h5')

###################### Predict from webcam ########################################"
video = cv2.VideoCapture(0)

#test_datagen = image.ImageDataGenerator(rescale=1./255)
test_datagen = image.ImageDataGenerator(rescale=1.0 / 255.0, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

while True:
    _, frame = video.read()
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
            if prediction[0] > 0.5:
                label = "plastic " + str(prediction[0])
                df = coordinates.iloc[i]
                x = df['x']
                y = df['y']
                w = df['w']
                h = df['h']
                cv2.putText(frame, label, (x, y+10), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 2)
                cv2.drawContours(frame, [my_contours[i]], -1, (255, 255, 255), 3)
            i += 1

    cv2.imshow("plastic detection", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()

########################## Predict one image ###########################################
# img1 = load_img('plastic342.jpg')
# img1 = img1.resize((250,250))
# img1 = img_to_array(img1)
#
# #img = Image.open('plastic342.jpg').resize((250, 250))
# test_datagen = image.ImageDataGenerator(rescale=1./255)
# img = test_datagen.standardize(np.copy(img1))
# #img = Image.open('test2.jpg').resize((250, 250))
# img_array = np.expand_dims(np.asarray(img), axis=0)
# print("img_array.size: ", img_array.shape)
# print("img_array: ", img_array)
#
# prediction = model.predict(img_array)
#
# print("prediction: ", prediction)




############################ Predict a directory ##############################################
# BASE_DIR = Path().resolve().parent
# waste = BASE_DIR / "detection_waste"
# train_data = waste / "train"
# test_data = waste / "test"
# print('test_data: ', test_data)
#
# test_datagen = image.ImageDataGenerator(rescale=1./255)
# img_batch = test_datagen.flow_from_directory(test_data, target_size=(250, 250), shuffle=False)
#
# data_list = []
# batch_index = 0
# while batch_index <= img_batch.batch_index:
#     data = img_batch.next()
#     data_list.append(data[0])
#     batch_index = batch_index + 1
#
# #print('data_list: ', data_list)
# predictions = model.predict(img_batch)#, steps=1)
#
# filenames = img_batch.filenames
# #print('filenames: ', filenames)
# predictions1D = np.asarray(predictions).flatten()
# #print('predictions: ', predictions1D)
# results = pd.DataFrame({"Filename" : filenames,
#                       "Prediction" : predictions1D})
#
# print('Results: ', results)
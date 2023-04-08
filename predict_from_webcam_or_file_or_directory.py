import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from PIL import Image
from keras.preprocessing import image
from pathlib import Path

from keras.utils import load_img, img_to_array

model = load_model('detection_waste_plastic.h5')

###################### Predict from webcam ########################################"
video = cv2.VideoCapture(0)

test_datagen = image.ImageDataGenerator(rescale=1./255)
while True:
    _, frame = video.read()

    im = Image.fromarray(frame, 'RGB')

    im = im.resize((250, 250))
    im1 = img_to_array(im)
    img = test_datagen.standardize(np.copy(im1))
    img_array = np.expand_dims(np.asarray(img), axis=0)

    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        label = "plastic " + str(prediction[0])
    else:
        label = "other_waste " + str(prediction[0])

    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow('plastic detection', frame)

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
# test_data = train_data / "test"
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
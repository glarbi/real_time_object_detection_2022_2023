import glob
import os

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping
from tensorflow import keras
import seaborn as sns
from keras.models import Sequential,load_model
from keras.layers import LeakyReLU,Conv2D,Activation, MaxPooling2D,Dense,Flatten,ReLU
import matplotlib.pyplot as plt_False_Positive_vs_True_Positive
from keras.utils import load_img, img_to_array
from sklearn.metrics import accuracy_score,roc_curve,confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score
from keras.preprocessing.image import ImageDataGenerator


BASE_DIR = Path().resolve().parent
waste = BASE_DIR / "detection_waste"
waste.mkdir(exist_ok= True,parents=True)
train_data = waste / "train"
train_data.mkdir(exist_ok= True,parents=True)
test_data = train_data / "test"
test_data.mkdir(exist_ok= True,parents=True)
train_data
classes=('others_waste','plastic')


#Build CNN Model
#frst
model = keras.Sequential()
model.add(keras.Input(shape=(250, 250, 3)))  # 250x250x3 RGB images
model.add(keras.layers.Conv2D(32, 5, strides=2, use_bias=False))  # output: 123x123x32
model.add(keras.layers.BatchNormalization(
    axis=3))  # Axis=3 "channels" is the 3rd axis of default data_format(batch_size, height, width, channels)
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Conv2D(64, 3, use_bias=False))  # output: 121x121x64
model.add(keras.layers.BatchNormalization(axis=3))
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.MaxPooling2D(3))  # output: 40x40x64

model.add(keras.layers.Conv2D(128, 3, use_bias=False))  # output: 38x38x128
model.add(keras.layers.BatchNormalization(axis=3))
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.MaxPooling2D(3))  # output: 12x12x128

# Classification layer.
model.add(keras.layers.Flatten())  # 1x1x18432
model.add(keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'))  # 1x1x128
model.add(keras.layers.Dense(1, activation='sigmoid'))  # 1x1x1
# compile model
opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)  # w = w + momentum * velocity - lr * grad
#model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])


# compile model
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

#  Display model
model.summary()
#####################
# create data generator
datagen = ImageDataGenerator(rescale=1.0/255.0, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
# prepare iterators
train_it = datagen.flow_from_directory('train', class_mode='binary', batch_size=64, target_size=(250, 250))
test_it = datagen.flow_from_directory('test', class_mode='binary', batch_size=64, target_size=(250, 250))
# fit model
es = EarlyStopping(monitor='val_accuracy', mode='max', patience=20)
history = model.fit(train_it, steps_per_epoch=len(train_it), validation_data=test_it, validation_steps=len(test_it), epochs=2, verbose=1, callbacks=[es], workers=10)

#  "Accuracy"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# save model
model.save('detection_waste_plastic.h5')



#Evaluate CNN Model

# load and prepare the image
def load_image(filename):
	# load the image
  img = load_img(filename, target_size=(250, 250))
  # convert to array
  img = img_to_array(img)
  print('[INFO] ',img.shape)
  # reshape into a single sample with 3 channels
  img = img.reshape(1, 250, 250, 3)
  print('[INFO] ',img.shape)
  # center pixel data
  img = img.astype('float32')
  img = img/255.0  # Scale pixel values to the range of 0-1
  return img

# load model
model = load_model('detection_waste_plastic.h5')

myFolder = waste / "test/plastic/2"
print("myFolder: ", myFolder)
filenames = os.listdir(myFolder)
print("filenames: ", filenames)

results = []
for arg in filenames:
  myFilename = "{}/{}".format(myFolder, arg)
  # load the image
  print('[INFO] Loading image: ', myFilename)
  img = load_image(myFilename)
  # predict the class
  print('[INFO] Predicting for ', myFilename)
  result = model.predict(img)
  results.append(result[0][0])
print("[INFO] Results: ", results)

y_true=('others_waste','plastic')


#y_pred_prob = y_pred[:,1]
y_pred_prob = results

y_pred_binary =  y_pred_prob > 0.5

#Confusion Matrix
print('\nConfusion Matrix\n -------------------------')
print(confusion_matrix(y_true,y_pred_binary))

sns.heatmap(confusion_matrix(y_true,y_pred_binary)/np.sum(confusion_matrix(y_true,y_pred_binary)), annot=True,
        fmt='.2%', cmap='Blues')

# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(y_true, y_pred_binary)
print('Accuracy: %f' % accuracy)


# precision tp / (tp + fp)
precision = precision_score(y_true, y_pred_binary)
print('Precision: %f' % precision)

# recall: tp / (tp + fn)
recall = recall_score(y_true, y_pred_binary)
print('Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_true, y_pred_binary)
print('F1 score: %f' % f1)

# ROC AUC
auc = roc_auc_score(y_true, y_pred_prob)
print('ROC AUC: %f' % auc)


# calculate roc curves
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)

# plot the roc curve for the model
plt.figure()
plt_False_Positive_vs_True_Positive.plot(fpr, tpr, linestyle='--', label='')

####################################################################################################
####################################################################################################

# import numpy as np
# from pathlib import Path
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import seaborn as sns
# from keras.models import Sequential, load_model
# from keras.layers import LeakyReLU, Conv2D, Activation, MaxPooling2D, Dense, Flatten, ReLU
# import matplotlib.pyplot as plt_False_Positive_vs_True_Positive
# from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, precision_score, recall_score, f1_score, \
#     roc_auc_score
# from keras.callbacks import EarlyStopping
# from keras import optimizers
# from keras.preprocessing.image import ImageDataGenerator
#
# BASE_DIR = Path().resolve().parent
# waste = BASE_DIR / "detection_waste"
# waste.mkdir(exist_ok=True, parents=True)
# train_data = waste / "train"
# train_data.mkdir(exist_ok=True, parents=True)
# test_data = train_data / "test"
# test_data.mkdir(exist_ok=True, parents=True)
# train_data
# classes = ('others_waste', 'plastic')
#
# # Build CNN Model
# # frst
# model = Sequential()
# model.add(Conv2D(8, (5, 5), padding='same', input_shape=(256, 256, 3), activation='relu'))
# model.add(LeakyReLU(alpha=0.1))
# model.add(MaxPooling2D((2, 2)))
#
# # scnd
# model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
# model.add(LeakyReLU(alpha=0.1))
# model.add(MaxPooling2D((2, 2)))
#
# #  Flatten
# model.add(Flatten())
# # Add a Dense layer with 512 unit
# model.add(Dense(512, activation='softmax'))
# model.add(LeakyReLU(alpha=0.1))
#
# # Add a final Dense layer
# model.add(Dense(2, activation='softmax'))
#
# #  Display model
# model.summary()
#
# # compile model
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# #####################
#
# # Train CNN model
# train_datagen = ImageDataGenerator(
#     rescale=1.0 / 255.0,
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=10,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     zoom_range=0.2,
#     brightness_range=[0.2, 1.0],
#     validation_split=0.3
# )
#
# train_image = train_datagen.flow_from_directory('train')
# target_size = (64, 64)
# batch_size = 32
#
# test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
#
# tst_image = test_datagen.flow_from_directory('test')
# target_size = (64, 64)
# batch_size = 32
#
# es = EarlyStopping(monitor='val_accuracy', mode='max', patience=20)
#
# history = model.fit(train_image, steps_per_epoch=len(train_image), validation_data=tst_image,
#                     validation_steps=len(tst_image), epochs=2, callbacks=[es], verbose=1, workers=10)
#
# #  "Accuracy"
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
#
# # "Loss"
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
#
# # save model
# model.save('detection_waste_plastic.h5')
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
#
# # Evaluate CNN Model
#
#
# # load model
# model = load_model('detection_waste_plastic.h5')
#
# # load test data
# batch_size = 32
# test_datagen = ImageDataGenerator(
#     rescale=1.0 / 255.0,
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     validation_split=0.2)
#
# test_it = test_datagen.flow_from_directory(train_data, classes,
#                                            shuffle=False, batch_size=batch_size, target_size=(256, 256),
#                                            subset='validation')
#
# y_true = test_it.classes
#
# y_pred = model.predict(test_it, steps=len(test_it), verbose=1)
#
# y_pred_prob = y_pred[:, 1]
#
# y_pred_binary = y_pred_prob > 0.5
#
# # Confution Matrix
# print('\nConfusion Matrix\n -------------------------')
# print(confusion_matrix(y_true, y_pred_binary))
#
# sns.heatmap(confusion_matrix(y_true, y_pred_binary) / np.sum(confusion_matrix(y_true, y_pred_binary)), annot=True,
#             fmt='.2%', cmap='Blues')
#
# # accuracy: (tp + tn) / (p + n)
# accuracy = accuracy_score(y_true, y_pred_binary)
# print('Accuracy: %f' % accuracy)
#
# # precision tp / (tp + fp)
# precision = precision_score(y_true, y_pred_binary)
# print('Precision: %f' % precision)
#
# # recall: tp / (tp + fn)
# recall = recall_score(y_true, y_pred_binary)
# print('Recall: %f' % recall)
#
# # f1: 2 tp / (2 tp + fp + fn)
# f1 = f1_score(y_true, y_pred_binary)
# print('F1 score: %f' % f1)
#
# # ROC AUC
# auc = roc_auc_score(y_true, y_pred_prob)
# print('ROC AUC: %f' % auc)
#
# # calculate roc curves
# fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
#
# # plot the roc curve for the model
# plt.figure()
# plt_False_Positive_vs_True_Positive.plot(fpr, tpr, linestyle='--', label='')
#











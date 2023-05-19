import cv2 
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import pandas as pd
from tensorflow import keras
import seaborn as sns
from keras.models import Sequential,load_model
from keras.layers import LeakyReLU,Conv2D,Activation, MaxPooling2D,Dense,Flatten,ReLU
import matplotlib.pyplot as plt_False_Positive_vs_True_Positive
from sklearn.metrics import accuracy_score,roc_curve,confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from keras.preprocessing import image
from keras.utils import load_img, img_to_array
from collections import deque




BASE_DIR = Path().resolve().parent
waste = BASE_DIR / "detection_waste"
waste.mkdir(exist_ok= True,parents=True)
train_data = waste / "train"
train_data.mkdir(exist_ok= True,parents=True)
test_data = train_data / "test"
test_data.mkdir(exist_ok= True,parents=True)
train_data



#Build CNN Model
#1
model = keras.Sequential()
model.add(keras.Input(shape=(250, 250, 3)))  # 250x250x3 RGB images
model.add(keras.layers.Conv2D(32, 5, strides=2, use_bias=False))  # output: 123x123x32
model.add(keras.layers.BatchNormalization(axis=3))  # Axis=3 "channels" is the 3rd axis of default data_format(batch_size, height, width, channels)
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

opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)  # w = w + momentum * velocity - lr * grad

# compile model
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

#  Display model
model.summary()



#####################

# create data generator
datagen = ImageDataGenerator(rescale=1.0/255.0,
                             
                             validation_split=0.3)

# prepare iterators

train_it = datagen.flow_from_directory('train', class_mode='binary', batch_size=32, target_size=(250, 250),subset='training')
valid_it= datagen.flow_from_directory('train', class_mode='binary', batch_size=32, target_size=(250, 250),subset='validation')



# fit model
es = EarlyStopping(monitor='val_accuracy', mode='max', patience=5)            
     
history = model.fit(train_it, steps_per_epoch=len(train_it), validation_data=valid_it,
                     validation_steps=len(valid_it), epochs=5, verbose=1, callbacks=[es], workers=10)

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

    
# load model
model = load_model('detection_waste_plastic.h5')

 
# load test data
tst_datagen = ImageDataGenerator(rescale=1.0/255.0, width_shift_range=0.1, height_shift_range=0.1, validation_split=0.2)

test_it= tst_datagen.flow_from_directory('train', class_mode='binary',shuffle=False
                                         , batch_size=32, target_size=(250, 250),subset='validation')


y_true = test_it.classes

y_pred = model.predict(test_it, steps=len(test_it), verbose=1)

y_pred_binary = y_pred> 0.5


#confusion Matrix    
print('\nConfusion Matrix\n -------------------------')    
print(confusion_matrix(y_true,y_pred))

sns.heatmap(confusion_matrix(y_true,y_pred)/np.sum(confusion_matrix(y_true,y_pred)), annot=True,
        fmt='.2%', cmap='Blues')

# accuracy: (tp + tn) / (p + n)

accuracy = accuracy_score(y_true, y_pred)
print('Accuracy: %f' % accuracy)


# precision tp / (tp + fp)
precision = precision_score(y_true, y_pred)
print('Precision: %f' % precision)

# recall: tp / (tp + fn)
recall = recall_score(y_true, y_pred)
print('Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_true,y_pred)
print('F1 score: %f' % f1)    
    
# ROC AUC
auc = roc_auc_score(y_true, y_pred)
print('ROC AUC: %f' % auc)


# calculate roc curves
fpr, tpr, _ = roc_curve(y_true, y_pred)
# plot the roc curve for the model
plt.figure()
plt_False_Positive_vs_True_Positive.plot(fpr, tpr, linestyle='--', label='')

# axis labels
plt_False_Positive_vs_True_Positive.xlabel('False Positive Rate')
plt_False_Positive_vs_True_Positive.ylabel('True Positive Rate')

        
# show the plot
plt_False_Positive_vs_True_Positive.show()




model = load_model('detection_waste_plastic.h5')
test_datagen = ImageDataGenerator(  rescale=1.0/255.0, width_shift_range=0.1, height_shift_range=0.1)
                                  
                                  
                              
    
test_it = test_datagen.flow_from_directory("test",class_mode='binary',shuffle=False, target_size=(250, 250))
probabilities = model.predict(test_it)

print(probabilities)     

y_pred = probabilities > 0.5

classit = [['plastic' for x in y_pred  if x[0]== True] ,['others_waste' for x in y_pred if x[0]==False]]
classit 
class_names = ['plastic', 'others_waste']
print(classit)



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


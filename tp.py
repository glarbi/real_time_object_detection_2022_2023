import cv2 
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from keras.models import Sequential,load_model
from keras.layers import LeakyReLU,Conv2D,Activation, MaxPooling2D,Dense,Flatten,ReLU
import matplotlib.pyplot as plt_False_Positive_vs_True_Positive
from sklearn.metrics import accuracy_score,roc_curve,confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score
from keras.callbacks import EarlyStopping
from keras import optimizers
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
model = Sequential()
model.add(Conv2D(8, (5, 5), padding='same', input_shape=(256, 256, 3),activation='relu')) 
model.add(LeakyReLU(alpha=0.1)) 
model.add(MaxPooling2D((2, 2)))

#scnd
model.add(Conv2D(8, (3, 3),  padding='same',activation='relu'))  
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2)))

#  Flatten 
model.add(Flatten())
# Add a Dense layer with 512 unit
model.add(Dense(512,activation='softmax')) 
model.add(LeakyReLU(alpha=0.1))

# Add a final Dense layer
model.add(Dense(2,activation='softmax'))

#  Display model
model.summary()

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#####################

#Train CNN model
train_datagen = ImageDataGenerator(
                                     rescale=1.0/255.0,
                                     featurewise_center= True,
                                     featurewise_std_normalization = True,
                                     rotation_range=10,
                                     width_shift_range=0.1,
                                     height_shift_range=0.1,
                                     zoom_range=0.2,                                     
                                     brightness_range=[0.2,1.0],
                                     validation_split=0.3
                                     )

 


train_image=train_datagen.flow_from_directory('train')
target_size=(64,64)
batch_size=32


test_datagen = ImageDataGenerator(rescale=1.0/255.0)

tst_image= test_datagen.flow_from_directory('test')
target_size=(64,64)
batch_size=32



es = EarlyStopping(monitor='val_accuracy', mode='max', patience=20)


history = model.fit(train_image, steps_per_epoch=len(train_image),validation_data=tst_image, 
                                  validation_steps=len(tst_image), epochs=20, callbacks=[es], verbose=1 ,workers =10 )



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
converter = tf.lite.TFLiteConverter.from_keras_model(model)



#Evaluate CNN Model

    
# load model
model = load_model('detection_waste_plastic.h5')

 
# load test data
batch_size=32
test_datagen = ImageDataGenerator(
                                    rescale=1.0/255.0,
                                    featurewise_center= True,
                                    featurewise_std_normalization = True,
                                    validation_split=0.2)

test_it = test_datagen.flow_from_directory(train_data,classes,
                                            shuffle=False,batch_size=batch_size, target_size=(256, 256),subset='validation')

y_true = test_it.classes

y_pred = model.predict(test_it, steps=len(test_it), verbose=1)


y_pred_prob = y_pred[:,1]

    
y_pred_binary =  y_pred_prob > 0.5

#Confution Matrix    
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














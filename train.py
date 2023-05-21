from pathlib import Path
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

BASE_DIR = Path().resolve().parent
# waste = BASE_DIR / "detection_waste/out"
waste = BASE_DIR / "detection_waste/shapes"
train_data = waste / "train"
test_data = waste / "test"
print('train_data :', train_data)
classes = ('other', 'plastic')

# Build CNN Model
# frst
model = keras.Sequential()
model.add(keras.Input(shape=(250, 250, 3)))  # 250x250x3 RGB images
# model.add(keras.Input(shape=(250, 250, 1)))  # 250x250x3 RGB images
model.add(keras.layers.Conv2D(32, 5, strides=2, use_bias=False))  # output: 123x123x32
model.add(keras.layers.BatchNormalization(axis=3))  # Axis=3 "channels" is the 3rd axis of default data_format
# (batch_size, height, width, channels)
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
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

#  Display model
model.summary()

##################### Train the model ################################
# create data generator
datagen = ImageDataGenerator(rescale=1.0 / 255.0, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
# prepare iterators
train_it = datagen.flow_from_directory(train_data, class_mode='binary', batch_size=64, target_size=(250, 250))
test_it = datagen.flow_from_directory(test_data, class_mode='binary', batch_size=64, target_size=(250, 250))
# train_it = datagen.flow_from_directory(train_data, color_mode='grayscale', class_mode='binary', batch_size=64, target_size=(250, 250))
# test_it = datagen.flow_from_directory(test_data, color_mode='grayscale', class_mode='binary', batch_size=64, target_size=(250, 250))
# fit model
es = EarlyStopping(monitor='val_accuracy', mode='max', patience=20)
history = model.fit(train_it, steps_per_epoch=len(train_it), validation_data=test_it, validation_steps=len(test_it),
                    epochs=100, verbose=1, callbacks=[es], workers=10)

##################### Test the model ######################
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

# Save the model
model.save('shape_based_detection_bottle_plastic_100epochs.h5')

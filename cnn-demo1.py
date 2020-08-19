import os
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import tensorflow as tf
import zipfile
import random
import shutil
from keras.optimizers import RMSprop
from shutil import copyfile



train = sum(len(files) for _, _, files in os.walk(r'Dog-Cat-Data/train'))
test = sum(len(files) for _, _, files in os.walk(r'Dog-Cat-Data/test'))

# print('Number of train images: {} \nNumber of test images: {} '.format(train,test))

fig, axs = plt.subplots(2, 5, figsize=(100, 100))
count = 0
# for every class in the dataset
for i in os.listdir('./Dog-Cat-Data/train'):
    # get the list of all images that belong to a particular class
    train_class = os.listdir(os.path.join('Dog-Cat-Data/train', i))

    # plot 5 images per class
    for j in range(5):
        img = os.path.join('Dog-Cat-Data/train', i, train_class[j])
        axs[count][j].title.set_text(i)
        axs[count][j].imshow(Image.open(img))
    count += 1
fig.tight_layout()
# plt.show()

No_images_per_class = []
Class_name = []
for i in os.listdir('./Dog-Cat-Data/train'):
  Class_name.append(i)
  train_class = os.listdir(os.path.join('Dog-Cat-Data/train',i))
  print('Number of images in {}={}\n'.format(i,len(train_class)))
  No_images_per_class.append(len(train_class))

fig = plt.figure(figsize=(10,5))
plt.bar(Class_name, No_images_per_class, color = sns.color_palette("cubehelix",len(Class_name)))
fig.tight_layout()
# plt.show()

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('Dog-Cat-Data/train',
                                                 target_size = (256, 256),
                                                 batch_size = 10,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('Dog-Cat-Data/test',
                                            target_size = (256, 256),
                                            batch_size = 10,
                                            class_mode = 'binary')

model = Sequential()

model.add(Conv2D(16, (3, 3), input_shape = (256, 256, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(units = 512, activation = 'relu'))

model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
'''
history = model.fit_generator(training_set,
                              epochs=5,
                              verbose=1,
                              validation_data=test_set)
'''
# model.save("dog-cat-2.h5")
# print("Saved model to disk")

model.load_weights('dog-cat-2.h5')

# evaluate = model.evaluate_generator(test_set, steps = test_set.n // 32, verbose =1)

# print('Accuracy Test : {}'.format(evaluate[1]))

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('cat2.jpg', target_size = (256, 256))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
    print(prediction)
else:
    prediction = 'cat'
    print(prediction)

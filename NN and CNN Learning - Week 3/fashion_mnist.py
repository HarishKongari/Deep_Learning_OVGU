import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import random_rotation, random_shear, random_shift, random_zoom

mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_labels[0])
plt.imshow(train_images[0], cmap="Greys_r")

print(np.shape(train_images),
np.shape(test_images),
np.shape(train_labels),
np.shape(test_labels))
        
tuple_inputsize = (28, 28, 1)
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

layer_list = [tf.keras.layers.Convolution2D(filters = 32, kernel_size = (3, 3),
                               padding = 'same',input_shape = tuple_inputsize,
                               kernel_initializer = 
                               tf.initializers.RandomNormal(mean = 0.0,
                                stddev = 0.05, seed = None),
                             bias_initializer=tf.initializers.constant(0.001)),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
    tf.keras.layers.Convolution2D(filters = 28 , kernel_size = (3, 3),
                               padding = 'same',input_shape = tuple_inputsize,
                               kernel_initializer = 
                               tf.initializers.RandomNormal(mean = 0.0,
                                stddev = 0.05, seed = None),
                             bias_initializer=tf.initializers.constant(0.001)),
    tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
    tf.keras.layers.Flatten(),
              tf.keras.layers.Dense(686, activation=tf.nn.relu,
                           kernel_initializer = 
                               tf.initializers.RandomNormal(mean = 0.0,
                                stddev = 0.05, seed = None),
                             bias_initializer=tf.initializers.constant(0.001)),
              tf.keras.layers.Dense(686/2, activation=tf.nn.relu,
                                    kernel_initializer = 
                               tf.initializers.RandomNormal(mean = 0.0,
                                stddev = 0.05, seed = None),
                             bias_initializer=tf.initializers.constant(0.001)),
              tf.keras.layers.Dense(686/4, activation=tf.nn.relu,
                                    kernel_initializer = 
                               tf.initializers.RandomNormal(mean = 0.0,
                                stddev = 0.05, seed = None),
                             bias_initializer=tf.initializers.constant(0.001)),
              tf.keras.layers.Dense(10, activation = tf.nn.softmax)]

model = tf.keras.Sequential(layer_list)

train_images = train_images/255.0
test_images = test_images/255.0

augmented_image = []
augmented_image_labels = []

for num in range (0, train_images.shape[0]):

	for i in range(0, 1):
			# original image:
		augmented_image.append(train_images[num])
		augmented_image_labels.append(train_labels[num])

		if True:
			augmented_image.append(tf.keras.preprocessing.image.random_rotation(train_images[num], 20, row_axis=0, col_axis=1, channel_axis=2))
			augmented_image_labels.append(train_labels[num])

		if True:
			augmented_image.append(tf.keras.preprocessing.image.random_shear(train_images[num], 0.2, row_axis=0, col_axis=1, channel_axis=2))
			augmented_image_labels.append(train_labels[num])

		if True:
			augmented_image.append(tf.keras.preprocessing.image.random_shift(train_images[num], 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2))
			augmented_image_labels.append(train_labels[num])

		if True:
			augmented_image.append(tf.keras.preprocessing.image.random_zoom(train_images[num], (0.7, 0.7), row_axis=0, col_axis=1, channel_axis=2))
			augmented_image_labels.append(train_labels[num])



augmented_image = tf.convert_to_tensor(augmented_image)

augmented_image_labels = tf.convert_to_tensor(augmented_image_labels)

import numpy as np
augmented_image = np.asarray(augmented_image).reshape((-1, 28, 28, 1))

augmented_image_labels =  np.array(augmented_image_labels)

augmented_image_labels =  augmented_image_labels.reshape(120000,1)

#train_img_gen = datagen.flow(train_images, train_labels, batch_size=60)
tf.data.Dataset.from_tensor_slices(augmented_image)
tf.data.Dataset.from_tensor_slices(augmented_image_labels)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy', 'Precision', 'Recall'])

model.fit(augmented_image, augmented_image_labels, epochs = 3,
          validation_data = (test_images, test_labels), callbacks=callbacks)
model.evaluate(test_images, test_labels)
model.predict(test_images)

model.summary()

import pandas as pd
classes=np.arange(0, 10, 1)
y_pred=model.predict_classes(test_images)
con_mat = tf.math.confusion_matrix(labels=test_labels, predictions=y_pred).numpy()

con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
 
con_mat_df = pd.DataFrame(con_mat_norm,
                     index = classes, 
                     columns = classes)

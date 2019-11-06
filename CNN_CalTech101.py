'''Result: train accuracy : 93.5%
           test accuracy = 45.6%
           epochs = 5, batch_size = 32'''


import tensorflow as tf
print(tf.__version__)
from tensorflow import keras


import numpy as np
import matplotlib.pyplot as plt
from img_data_preprocessing import image_array, label_name


im_array = np.asarray(image_array, dtype=np.float32)

#im_array = image_array


from sklearn.model_selection import train_test_split
#mnist = tf.keras.datasets.mnist
train_images, test_images, train_labels, test_labels = train_test_split(
                                                           im_array, 
                                                           label_name, 
                                                          test_size = 1500)

# Encoding Categorical data( dependent data so no OneHotEncoder)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_train = LabelEncoder()
train_labels = labelencoder_train.fit_transform(train_labels)
train_labels = train_labels.reshape(-1,1)


test_labels = labelencoder_train.fit_transform(test_labels)
test_labels = test_labels.reshape(-1,1)

# Feature Scaling
'''Observation: After using StandardScaler instead
im_array = im_array/255.0, the training results improved
very much( 15 epochs 90% and test prediction also a bit better
about 40%(but still overfitting)'''
from sklearn.preprocessing import StandardScaler
scaler = {}
for i in range(train_images.shape[1]):
    scaler[i] = StandardScaler()
    train_images[:, i, :] = scaler[i].fit_transform(train_images[:, i, :])
    
for i in range(test_images.shape[1]):
    test_images[:, i, :] = scaler[i].fit_transform(test_images[:, i, :])








plt.imshow(train_images[0], cmap="Greys_r")




# just set up a "chain" of hidden layers
'''layers = []
for layer in range(n_layers):
    layers.append(tf.keras.layers.Dense(
        n_units, activation=tf.nn.relu,
        kernel_initializer=tf.initializers.RandomNormal(mean = 0.0,
                                                        stddev = 0.05,
                                                        seed = None),
        bias_initializer=tf.initializers.constant(0.001)))

# finally add the output layer
model.append(tf.keras.layers.Dense(
    102, kernel_initializer=tf.initializers.RandomNormal(mean = 0.0,
                                                        stddev = 0.05,
                                                        seed = None)))'''

'''When there was only 1 hidden layer, test was about 40% but it needed 
more epochs (200) for reaching training 70%, after adding multiple hidden
layers of Dense(Artificial NN) then training reached 90% with epochs = 50%
and test = 43.5% with batch_size = 64 

Flatten had input size of (64,64) before Convolution2D input size was added'''
'''inputsize = list(train_images.shape)
inputsize.append(1)
tuple_inputsize = tuple(inputsize)'''

'''Only for giving input_shape to Convolution_2D, had to create 
tuple_inputsize to (64, 64, 1) instead (64, 64)
Had to reshape train_images and test_images becoz there was error
saying ndim = 4 is expected'''
tuple_inputsize = (64, 64, 1)
train_images = train_images.reshape(-1, 64, 64, 1)
test_images = test_images.reshape(-1, 64, 64, 1)

model = keras.Sequential([
    keras.layers.Convolution2D(filters = 64, kernel_size = (3, 3),
                               padding = 'same',input_shape= tuple_inputsize,
                               kernel_initializer = 
                               tf.initializers.RandomNormal(mean = 0.0,
                                stddev = 0.05, seed = None),
                             bias_initializer=tf.initializers.constant(0.001)),                                                        
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Convolution2D(filters = 32, kernel_size = (3, 3),
                               padding = 'same',input_shape= tuple_inputsize,
                               kernel_initializer = 
                               tf.initializers.RandomNormal(mean = 0.0,
                                stddev = 0.05, seed = None),
                             bias_initializer=tf.initializers.constant(0.001)),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(1000, activation='relu'),
    keras.layers.Dense(600, activation='relu'),
    keras.layers.Dense(250, activation='relu'),
    keras.layers.Dense(102, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])







model.fit(train_images, train_labels, batch_size = 32, epochs = 5 )


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\n Test accuracy:', test_acc)

predictions = model.predict(test_images[:3])
print('\n predictions shape:', predictions.shape)

np.argmax(predictions[0])

test_labels[0]




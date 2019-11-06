
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

#Image_Path = "C:\\Soumick\\Prostate Classification task\\train1"
Image_Path = "C:\\Courses\\Selected Topics in Image Understanding\\101_ObjectCategories"
os.chdir(Image_Path)
list_fams = os.listdir(os.getcwd()) # vector of strings with family names
list_fams = sorted(list_fams, key=str.lower) #to ensure that the file names is sorted

img_files_list = [] 
labels_img_list = [] 
label_name = []
for directory, sub_directory, image_list in os.walk(Image_Path):
    #print(directory)
    #print(image_list)
    label_img_list = directory.split('\\')
    for image_name in image_list:
        if ".jpg" in image_name.lower():  # check whether the file's DICOM
            img_files_list.append(os.path.join(directory,image_name))
            #labels_img_list.append(label_name)
            label_name.append(label_img_list[-1])
    
# Get ref file
sample_ds = cv2.imread(img_files_list[0], 0)
plt.imshow(sample_ds, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()


# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
check_pixel_dim = (int(len(sample_ds[0])), int(len(sample_ds[1])), len(img_files_list))

'''# Load spacing values (in mm)
check_pixel_spacing = (float(sample_ds.PixelSpacing[0]), float(sample_ds.PixelSpacing[1]), float(sample_ds.SliceThickness))'''

# The array is sized based on 'check_pixel_dim'
image_array = []


# loop through all the DICOM files
for image in img_files_list:
    # read the file
    ds = cv2.imread(image, 0)
    
    img = cv2.resize(ds, (96, 96))
    
    #image_array[:, :] = img
    image_array.append(img)
'''im_array = np.asarray(image_array, dtype=np.float32)  '''  

        
            
    

    #cv2.imwrite(os.path.join(path_g , "img_gray_%d.jpg" %x), ct101_gray[x])
#cv2.waitKey(0)
     
        
    # store the raw image data
   # image_array[:, :, img_files_list.index(image)] = ds.pixel_array





import tensorflow as tf
print(tf.__version__)
from tensorflow import keras


#import numpy as np
#import matplotlib.pyplot as plt
'''from img_data_preprocessing import image_array, label_name, list_fams'''

from keras.preprocessing.image import ImageDataGenerator


train_image_array =[]
test_image_array = [] 
train_label_name = []
test_label_name = []
count = 0
count1 = 0   
for j in list_fams:
    count1 = count + count1
    count = label_name.count(j)
    
    for jj in range(int(0.8 * count)):
        train_image_array.append(image_array[jj + count1])
        train_label_name.append(label_name[jj + count1])
    for jjj in range(count - int(0.8 * count)):
        test_image_array.append(image_array[jj + jjj + count1])
        test_label_name.append(label_name[jj + count1])

train_image_array = np.asarray(train_image_array, dtype=np.float64) 
test_image_array = np.asarray(test_image_array, dtype=np.float64) 
#im_array = image_array


'''from sklearn.model_selection import train_test_split
#mnist = tf.keras.datasets.mnist
train_images, test_images, train_labels, test_labels = train_test_split(
                                                           im_array, 
                                                           label_name, 
                                                          test_size = 1500)'''

# Encoding Categorical data( dependent data so no OneHotEncoder)
from sklearn.preprocessing import LabelEncoder
labelencoder_train = LabelEncoder()
train_label_name = labelencoder_train.fit_transform(train_label_name)
train_label_name = train_label_name.reshape(-1,1)


test_label_name = labelencoder_train.fit_transform(test_label_name)
test_label_name = test_label_name.reshape(-1,1)

# Feature Scaling
'''Observation: After using StandardScaler instead
im_array = im_array/255.0, the training results improved
very much( 15 epochs 90% and test prediction also a bit better
about 40%(but still overfitting)'''
from sklearn.preprocessing import StandardScaler
scaler = {}
for i in range(train_image_array.shape[1]):
    scaler[i] = StandardScaler()
    train_image_array[:, i, :] = scaler[i].fit_transform(train_image_array[:, i, :])
    
for i in range(test_image_array.shape[1]):
    test_image_array[:, i, :] = scaler[i].fit_transform(test_image_array[:, i, :])


'''faced problem plotting train_images[0], because the shape of image has 
changed from (64, 64) to (64, 64, 1)'''
#plt.imshow(train_images[0], cmap="Greys_r")



'''this two lines had to be brought from down to up because there was error
occuring at train_datagen.fit shape problem'''
'''train_images = train_images.reshape(-1, 64, 64, 1)
test_images = test_images.reshape(-1, 64, 64, 1)



train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True, data_format = 'channels_first')

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
train_datagen.fit(train_images)'''







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
tuple_inputsize = (96, 96, 1)
train_image_array = train_image_array.reshape(-1, 96, 96, 1)
test_image_array = test_image_array.reshape(-1, 96, 96, 1)
'''train_images = train_images.reshape(-1, 64, 64, 1)
test_images = test_images.reshape(-1, 64, 64, 1)'''


train_datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True, data_format = 'channels_last')

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
train_datagen.fit(train_image_array)

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



# fits the model on batches with real-time data augmentation:
'''model.fit_generator(train_datagen.flow(train_images, train_labels, batch_size=32),
                    steps_per_epoch=len(train_images) / 64, epochs= 10)'''
model.fit_generator(train_datagen.flow(train_image_array, train_label_name, batch_size=32),
                    steps_per_epoch=len(train_image_array) / 256, epochs= 25)

# here's a more "manual" example
train_img_gen = []
train_label_gen =[]

for e in range(5):
    print('Epoch', e)
    batches = 0
    
    for x_batch, y_batch in train_datagen.flow(train_image_array, train_label_name, batch_size=32):
        train_img_gen.append(x_batch)
        train_label_gen.append(y_batch)
        
        
        model.fit(x_batch, y_batch, epochs = 5)
        batches += 1
        if batches >= len(train_image_array) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break



'''train_img_gen_1 = np.asarray(train_img_gen, dtype=np.uint8)
train_label_gen_1 = np.asarray(train_label_gen, dtype=np.uint8)



#model.fit(train_images, train_labels, batch_size = 64, epochs = 10 )
for e in train_img_gen:
    model.fit(train_img_gen[e], train_label_gen[e], batch_size = 64, epochs = 10 )'''


'''model.fit(train_image_array, train_label_name, batch_size = 64, epochs = 10 )'''


test_loss, test_acc = model.evaluate(test_image_array,  test_label_name, verbose=2)

print('\n Test accuracy:', test_acc)

predictions = model.predict(test_image_array[:3])
print('\n predictions shape:', predictions.shape)

np.argmax(predictions[0])


test_labels[0]




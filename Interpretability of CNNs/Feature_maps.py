import tensorflow as tf
import keras
from matplotlib import pyplot
# load vgg model
from keras.applications.vgg16 import VGG16
# load the model
model = tf.keras.applications.vgg16.VGG16()
# summarize the model
model.summary()

# summarize filter shapes
for layer in model.layers:
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
    
# get filter weights
filters, biases = layer.get_weights()
print(layer.name, filters.shape)


    

# retrieve weights from the second hidden layer
filters, biases = model.layers[1].get_weights()



# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)



# plot first few filters
n_filters, ix = 6, 1
for i in range(n_filters):
	# get the filter
	f = filters[:, :, :, i]
	# plot each channel separately
	for j in range(3):
		# specify subplot and turn of axis
		ax = pyplot.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(f[:, :, j], cmap='gray')
		ix += 1
# show the figure
pyplot.show()



# summarize feature map size for each conv layer

from matplotlib import pyplot
# load the model

# summarize feature map shapes
for i in range(len(model.layers)):
	layer = model.layers[i]
	# check for convolutional layer
	if 'conv' not in layer.name:
		continue
	# summarize output shape
	print(i, layer.name, layer.output.shape)
    
from keras.models import Model
# redefine model to output right after the first hidden layer
model = tf.keras.models.Model(inputs=model.inputs, outputs=model.layers[1].output)


# load the image with the required shape

# visualize feature maps output from each block in the vgg model
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from matplotlib import pyplot
from numpy import expand_dims
# load the model
model = tf.keras.applications.vgg16.VGG16()
# redefine model to output right after the first hidden layer
ixs = [2, 5, 9, 13, 17]
outputs = [model.layers[i].output for i in ixs]
model = tf.keras.models.Model(inputs=model.inputs, outputs=outputs)
# load the image with the required shape
img = load_img('aeroplane.jpg', target_size=(224, 224))
# convert the image to an array
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = model.predict(img)
# plot the output from each block
square = 8
for fmap in feature_maps:
	# plot all 64 maps in an 8x8 squares
	ix = 1
	for _ in range(square):
		for _ in range(square):
			# specify subplot and turn of axis
			ax = pyplot.subplot(square, square, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			# plot filter channel in grayscale
			pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray')
			ix += 1
	# show the figure
	pyplot.show()









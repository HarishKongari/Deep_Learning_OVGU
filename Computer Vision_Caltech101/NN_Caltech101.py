import tensorflow as tf
print(tf.__version__)


import numpy as np
import matplotlib.pyplot as plt
from img_data_preprocessing import image_array


from sklearn.model_selection import train_test_split
#mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = train_test_split(
                                                           image_array, 
                                                           label_name, 
                                                           test_size = 0.2, 
                                                           random_state = 0)

plt.imshow(train_images[0], cmap="Greys_r")

data = MNISTDataset(train_images.reshape([-1, 784]), train_labels, 
                    test_images.reshape([-1, 784]), test_labels,
                    batch_size=128)


train_steps = 2000
learning_rate = 0.1
coefficient = 0.5


#Initializing variables to 0 will not work for multilayer perceptrons:
#When the weights are initialized to 0 , all outputs of the hidden layer are the 
#same , so the gradient of the loss function is the same for each weight , and the 
#weights will have the same value in iterations.
#the profermance of the MLP is better at using random.normal 
#than with random.uniform(dont know why)
W1 = tf.Variable(tf.compat.v1.random.normal([784,500], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1,shape=[500]))

W2 = tf.Variable(np.zeros([500, 10]).astype(np.float32))
b2 = tf.Variable(np.zeros(10, dtype=np.float32))




for step in range(train_steps):
    img_batch, lbl_batch = data.next_batch()
    with tf.GradientTape() as tape:
        logits = tf.matmul(tf.nn.relu(tf.matmul(img_batch, W1) + b1),W2)+b2
        xent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
           logits=logits, labels=lbl_batch))
    grads = tape.gradient(xent, [W1,b1,W2, b2])
    W1.assign_sub(learning_rate * grads[0])
    b1.assign_sub(learning_rate * grads[1])
    W2.assign_sub(learning_rate * grads[2])
    b2.assign_sub(learning_rate * grads[3])
    
    if not step % 200:
        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(preds, lbl_batch),
                             tf.float32))
        print("Loss: {} Accuracy: {}".format(xent, acc))

        
test_preds = tf.argmax(tf.matmul(tf.nn.relu(tf.matmul(data.test_data, W1) + b1),W2)+b2, axis=1,
                       output_type=tf.int32)
acc = tf.reduce_mean(tf.cast(tf.equal(test_preds, data.test_labels),
                             tf.float32))
print(acc)



#Which of these parts do you think should be wrapped in higher-level interfaces?
#I think the part about gradient descent and applying gradients to variables 
#should be wrapped in higher-level interfaces
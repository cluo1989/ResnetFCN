# coding utf-8 
import cv2
import numpy as np 
import tensorflow as tf 
from resnet_fcn import inference
import matplotlib.pyplot as plt

NUM_CLASSES= 11
IMAGE_WIDTH = 224  # 256
IMAGE_HEIGHT = 224 # 384

x = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='input')
pred, logits = inference(x, is_training=False, num_classes=NUM_CLASSES, num_blocks=[3,4,6,3])

isess = tf.InteractiveSession()
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
ckpt_filename = './logs/model20500.ckpt-20501'
saver.restore(isess, ckpt_filename)

lab = cv2.imread('data/train/labels/label_000009.png')
lab = cv2.resize(lab, (224, 224))
img = cv2.imread('data/train/images/image_000009.png')/255.0
img = cv2.resize(img, (224, 224))
img = np.expand_dims(img, axis=0)  # np.reshape(img, (1, 224))
print(img.shape, img.dtype)
print(lab.shape, lab.dtype)

result = isess.run(pred, feed_dict={x: img})
result = np.squeeze(result, axis=(0, -1))
print(result.shape, result.dtype)

# ------------- show -------------
plt.figure(num='origin&result')
plt.subplot(131)
plt.title('origin')
plt.imshow(np.squeeze(img))
plt.axis('off')

plt.subplot(132)
plt.title('result')
plt.imshow(result)
plt.axis('off')

plt.subplot(133)
plt.title('label')
plt.imshow(lab[:,:,0])
plt.axis('off')

plt.show()

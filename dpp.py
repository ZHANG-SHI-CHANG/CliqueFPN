import cv2
import numpy as np
import time

import tensorflow as tf

def lamda_1(x,lamda,eplice=1e-10):
    return tf.pow(tf.sqrt(tf.pow(x,2.)+eplice*eplice),lamda)
def lamda_2(x,lamda,eplice=1e-10):
    return tf.pow(tf.sqrt(tf.pow(tf.nn.relu(x),2.)+eplice*eplice),lamda)

Iq = tf.placeholder(tf.float32,[None,320,320,3])

avg_pool = tf.nn.avg_pool(Iq,[1,2,2,1],[1,2,2,1],'SAME')
max_pool = tf.nn.max_pool(Iq,[1,2,2,1],[1,2,2,1],'SAME')

Ip = tf.nn.avg_pool(Iq,[1,2,2,1],[1,1,1,1],'SAME')

gaussian_filter = tf.truediv(tf.constant([[1,1,1],[1,2,1],[1,1,1]],dtype=tf.float32),16.0)
gaussian_filter = tf.reshape(gaussian_filter,[3,3,1,1])
gaussian_filter = tf.tile(gaussian_filter,[1,1,Ip.get_shape().as_list()[-1],Ip.get_shape().as_list()[-1]])
Ip = tf.nn.conv2d(Ip,gaussian_filter,[1,1,1,1],'SAME')
#gaussian_filter = tf.tile(gaussian_filter,[1,1,Ip.get_shape().as_list()[-1],1])
#Ip = tf.nn.depthwise_conv2d(Ip,gaussian_filter,[1,1,1,1],'SAME')

alpha = tf.Variable(0.0,trainable=True)
lamda = tf.Variable(2.7,trainable=True)
weight = alpha+lamda_2(Iq-Ip,lamda)
inverse_bilatera = Iq*weight

result = tf.truediv(tf.nn.avg_pool(inverse_bilatera,[1,2,2,1],[1,2,2,1],'SAME'),tf.nn.avg_pool(weight,[1,2,2,1],[1,2,2,1],'SAME'))

with tf.Session() as sess:  
    img = cv2.imread('test.jpg')
    img = cv2.resize(img,(320,320)).astype(np.float)
    img = img[np.newaxis,:,:,:]
    
    sess.run(tf.global_variables_initializer())
    result,avg_pool,max_pool = sess.run([result,avg_pool,max_pool],feed_dict={Iq:img})
    
    cv2.namedWindow('avg_pool',cv2.WINDOW_NORMAL)
    cv2.namedWindow('max_pool',cv2.WINDOW_NORMAL)
    cv2.namedWindow('result',cv2.WINDOW_NORMAL)
    cv2.imshow('avg_pool',avg_pool[0,...].astype(np.uint8))
    cv2.imshow('max_pool',max_pool[0,...].astype(np.uint8))
    cv2.imshow('result',result[0,...].astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
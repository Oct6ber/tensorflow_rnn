import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./data/mnist', one_hot=True)
xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])
is_training = tf.placeholder(tf.bool, [])
def conv_wrapper(inputs,name,is_training,output_chanel,kernel_size,activation=tf.nn.relu,padding = 'same'):
    with tf.name_scope(name):
        conv2d = tf.layers.conv2d(inputs,output_chanel,kernel_size,activation=None,name = name+'/conv2d')
        bn = tf.layers.batch_normalization(conv2d,training=is_training)
        return activation(bn)

def pooling_wrapper(inputs,name):
    return tf.layers.max_pooling2d(inputs,(2,2),(2,2),padding='same',name = name)

x_image = tf.reshape(xs,[-1,28,28,1])
conv1 = conv_wrapper(x_image,'conv1',is_training,output_chanel=32,kernel_size=(5,5))
pooling1 = pooling_wrapper(conv1,'pooling1')

conv2 = conv_wrapper(pooling1,'conv2',is_training,output_chanel=64,kernel_size=(5,5))
pooling2 = pooling_wrapper(conv2,'pooling2')

fltten = tf.layers.flatten(pooling2)
d1 = tf.layers.dense(fltten,1024,activation=tf.nn.relu)
y_ = tf.layers.dense(d1,10,activation=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(y_), reduction_indices=[1]))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_,1),tf.argmax(ys,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch= mnist.train.next_batch(64)
        if i % 50 ==0:
            train_accuracy = accuracy.eval(feed_dict={xs:batch[0],ys:batch[1],is_training:True})
            data = "step %d, training accuracy %g"%(i,train_accuracy)
            print(data)
            resulttxt = str(data)
            with open('res.txt','a') as fi:
                fi.write(resulttxt)
                fi.write('\n')
        sess.run(train_op,feed_dict={xs:batch[0],ys:batch[1],is_training:True})
    data2 = "test accuracy %g"%accuracy.eval(feed_dict={xs:mnist.test.images,ys:mnist.test.labels,is_training:False})
    print(data2)
    re = str(data2)
    with open('res.txt', 'a') as fi:
        fi.write(re)
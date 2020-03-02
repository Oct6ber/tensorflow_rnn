import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./data/mnist', one_hot=True)

lr = 0.001
training_iters = 100000
batch_size = 128
n_inputs = 28 #当列
n_steps = 28  #当行
n_hidden_units = 128
n_classes = 10

weights = {
    #shape (28,128)
    'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    #shape (128,10)
    'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
}
biases = {
    #shape (128,)
    'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
    #shape (10,)
    'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
}
x= tf.placeholder(dtype=tf.float32,shape=[None,n_steps,n_inputs])
y= tf.placeholder(dtype=tf.float32,shape=[None,n_classes])

def RNN(X,weights,biases):
    #初始使x是三维数据  要做wx+b需要转换成二维，(128batch,28,28)->(128*28n_steps,28n_inputs)
    X = tf.reshape(X,[-1,n_inputs])
    X_in = tf.matmul(X,weights['in'])+biases['in']
    #需再转换为三维数据(128batch,28n_step,128hidden)
    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_units])

#如果 inputs 为 (batches, steps, inputs) ==> time_major=False;
# 如果 inputs 为 (steps, batches, inputs) ==> time_major=True;
#     state可被分为(c_state, h_state)
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1.0,state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)
    outputs,final_state = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=init_state,time_major=False)
    result = tf.matmul(final_state[1],weights['out']) + biases['out']
    # # 把 outputs 变成 列表 [(batch, outputs)..] * steps
    # outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out']  # 选取最后一个 output
    return result

pred = RNN(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
train = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step*batch_size < training_iters:
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape([batch_size,n_steps,n_inputs])
        sess.run(train,feed_dict={
            x:batch_x,
            y:batch_y
        })
        if step %20 == 0:
            print(sess.run(accuracy,feed_dict={
                x:batch_x,
                y:batch_y
            }))
        step +=1

# -*- coding:utf-8 -*-  
# ========================================================== #
# File name: NIN_tf.py
# Author: BIGBALLON
# Date created: 07/20/2017
# Python Version: 3.5.2
# Description: implement Network in Network only use tensorflow
#     Paper Link: (Network In Network) https://arxiv.org/abs/1312.4400
#      Trick Used:
#         Data augmentation parameters
#         Color normalization
#         Use Nesterov momentum
#         Weight Decay
#         He's Weight initialization [https://arxiv.org/abs/1502.01852]
#         Batch Normalization [https://arxiv.org/abs/1502.03167]
#         Use ELU instead of ReLu [https://arxiv.org/abs/1511.07289]
#        
# Result: Test accuracy about 91.5% 
# ========================================================== #

import tensorflow as tf
from data_utility import *

iterations      = 200
batch_size      = 250
total_epoch     = 164
weight_decay    = 0.0001
dropout_rate    = 0.5
momentum_rate   = 0.9
log_save_path   = './nin_bn_logs'
model_save_path = './model/'

# ========================================================== #
# ├─ weight_variable()  Kai-Ming He's WI sqrt(2/(k*k*c))
# ├─ bias_variable()
# ├─ conv2d()           With Batch Normalization
# ├─ max_pool()
# └─ global_avg_pool()
# ========================================================== #


def weight_variable(shape, stv=0.05):
    initial = tf.random_normal(shape, stddev=math.sqrt(2.0/(1.0*shape[3]*shape[1]*shape[0])), dtype=tf.float32 )
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32 )
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool(input, k_size=1, stride=1):
    return tf.nn.max_pool(input, ksize=[1, k_size, k_size, 1], strides=[1, stride, stride, 1], padding='SAME')

def global_avg_pool(input, k_size=1, stride=1):
    return tf.nn.avg_pool(input, ksize=[1,k_size,k_size,1], strides=[1,stride,stride,1], padding='VALID')

def batch_norm(input):
    return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3, is_training=train_flag, updates_collections=None)


# ========================================================== #
# ├─ _random_crop() 
# ├─ _random_flip_leftright()
# ├─ data_augmentation()
# ├─ data_preprocessing()
# └─ learning_rate_schedule()
# ========================================================== #


def _random_crop(batch, crop_shape, padding=None):
        oshape = np.shape(batch[0])
        
        if padding:
            oshape = (oshape[0] + 2*padding, oshape[1] + 2*padding)
        new_batch = []
        npad = ((padding, padding), (padding, padding), (0, 0))
        for i in range(len(batch)):
            new_batch.append(batch[i])
            if padding:
                new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                          mode='constant', constant_values=0)
            nh = random.randint(0, oshape[0] - crop_shape[0])
            nw = random.randint(0, oshape[1] - crop_shape[1])
            new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                                        nw:nw + crop_shape[1]]
        return new_batch

def _random_flip_leftright(batch):
        for i in range(len(batch)):
            if bool(random.getrandbits(1)):
                batch[i] = np.fliplr(batch[i])
        return batch

def data_preprocessing(x_train,x_test):

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train[:,:,:,0] = (x_train[:,:,:,0] - np.mean(x_train[:,:,:,0])) / np.std(x_train[:,:,:,0])
    x_train[:,:,:,1] = (x_train[:,:,:,1] - np.mean(x_train[:,:,:,1])) / np.std(x_train[:,:,:,1])
    x_train[:,:,:,2] = (x_train[:,:,:,2] - np.mean(x_train[:,:,:,2])) / np.std(x_train[:,:,:,2])

    x_test[:,:,:,0] = (x_test[:,:,:,0] - np.mean(x_test[:,:,:,0])) / np.std(x_test[:,:,:,0])
    x_test[:,:,:,1] = (x_test[:,:,:,1] - np.mean(x_test[:,:,:,1])) / np.std(x_test[:,:,:,1])
    x_test[:,:,:,2] = (x_test[:,:,:,2] - np.mean(x_test[:,:,:,2])) / np.std(x_test[:,:,:,2])

    return x_train, x_test

def learning_rate_schedule(epoch_num):
      if epoch_num < 81:
          return 0.1
      elif epoch_num < 121:
          return 0.01
      else:
          return 0.001

def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [32,32], 4)
    return batch

def run_testing(sess,ep):
    acc = 0.0
    loss = 0.0
    pre_index = 0
    add = 1000
    for it in range(10):
        batch_x = test_x[pre_index:pre_index+add]
        batch_y = test_y[pre_index:pre_index+add]
        pre_index = pre_index + add
        loss_, acc_  = sess.run([cross_entropy,accuracy],feed_dict={x:batch_x, y_:batch_y, keep_prob: 1.0, train_flag: False})
        loss += loss_ / 10.0
        acc += acc_ / 10.0
    summary = tf.Summary(value=[tf.Summary.Value(tag="test_loss", simple_value=loss), 
                            tf.Summary.Value(tag="test_accuracy", simple_value=acc)])
    return acc, loss, summary


# ========================================================== #
# ├─ main()
# Training and Testing 
# Save train/teset loss and acc for visualization
# Save Model in ./model
# ========================================================== #


if __name__ == '__main__':

    train_x, train_y, test_x, test_y = prepare_data()
    train_x, test_x = data_preprocessing(train_x, test_x)

    # define placeholder x, y_ , keep_prob, learning_rate
    x  = tf.placeholder(tf.float32,[None, image_size, image_size, 3])
    y_ = tf.placeholder(tf.float32, [None, class_num])
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    train_flag = tf.placeholder(tf.bool)

    # build_network

    W_conv1 = weight_variable([5, 5, 3, 192])
    b_conv1 = bias_variable([192])
    output  = tf.nn.elu( batch_norm(conv2d(x,W_conv1) + b_conv1))

    W_mlp11 = weight_variable([1, 1, 192, 160])
    b_mlp11 = bias_variable([160])
    output  = tf.nn.elu( batch_norm(conv2d(output,W_mlp11) + b_mlp11))

    W_mlp12 = weight_variable([1, 1, 160, 96])
    b_mlp12 = bias_variable([96])
    output  = tf.nn.elu( batch_norm(conv2d(output,W_mlp12) + b_mlp12))

    output  = max_pool(output, 3, 2)

    output  = tf.nn.dropout(output,keep_prob)

    W_conv2 = weight_variable([5, 5, 96, 192])
    b_conv2 = bias_variable([192])
    output  = tf.nn.elu( batch_norm(conv2d(output,W_conv2) + b_conv2))

    W_mlp21 = weight_variable([1, 1, 192, 192])
    b_mlp21 = bias_variable([192])
    output  = tf.nn.elu( batch_norm(conv2d(output,W_mlp21) + b_mlp21))

    W_mlp22 = weight_variable([1, 1, 192, 192])
    b_mlp22 = bias_variable([192])
    output  = tf.nn.elu( batch_norm(conv2d(output,W_mlp22) + b_mlp22))

    output  = max_pool(output, 3, 2)

    output  = tf.nn.dropout(output,keep_prob)

    W_conv3 = weight_variable([3, 3, 192, 192])
    b_conv3 = bias_variable([192])
    output  = tf.nn.elu( batch_norm(conv2d(output,W_conv3) + b_conv3))

    W_mlp31 = weight_variable([1, 1, 192, 192])
    b_mlp31 = bias_variable([192])
    output  = tf.nn.elu( batch_norm(conv2d(output,W_mlp31) + b_mlp31))

    W_mlp32 = weight_variable([1, 1, 192, 10])
    b_mlp32 = bias_variable([10])
    output  = tf.nn.elu( batch_norm(conv2d(output,W_mlp32) + b_mlp32))

    output  = global_avg_pool(output, 8, 1)

    output  = tf.reshape(output,[-1,10])


    # loss function: cross_entropy
    # train_step: training operation
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum_rate,use_nesterov=True).minimize(cross_entropy + l2 * weight_decay)
    
    correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    # initial an saver to save model
    saver = tf.train.Saver()
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_save_path,sess.graph)

        # epoch = 164 
        # batch size = 128
        # iteration = 391
        # we should make sure [bath_size * iteration = data_set_number]

        for ep in range(1,total_epoch+1):
            lr = learning_rate_schedule(ep)
            pre_index = 0
            train_acc = 0.0
            train_loss = 0.0
            start_time = time.time()

            print("\nepoch %d/%d:" %(ep,total_epoch))

            for it in range(1,iterations+1):
                batch_x = train_x[pre_index:pre_index+batch_size]
                batch_y = train_y[pre_index:pre_index+batch_size]

                batch_x = data_augmentation(batch_x)

                _, batch_loss = sess.run([train_step, cross_entropy],feed_dict={x:batch_x, y_:batch_y, keep_prob: dropout_rate, learning_rate: lr, train_flag: True})
                batch_acc = accuracy.eval(feed_dict={x:batch_x, y_:batch_y, keep_prob: 1.0, train_flag: True})

                train_loss += batch_loss
                train_acc  += batch_acc
                pre_index  += batch_size

                if it == iterations:
                    train_loss /= iterations
                    train_acc /= iterations

                    loss_, acc_  = sess.run([cross_entropy,accuracy],feed_dict={x:batch_x, y_:batch_y, keep_prob: 1.0, train_flag: True})
                    train_summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=train_loss), 
                                          tf.Summary.Value(tag="train_accuracy", simple_value=train_acc)])

                    val_acc, val_loss, test_summary = run_testing(sess,ep)

                    summary_writer.add_summary(train_summary, ep)
                    summary_writer.add_summary(test_summary, ep)
                    summary_writer.flush()

                    print("iteration: %d/%d, cost_time: %ds, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f" %(it, iterations, int(time.time()-start_time), train_loss, train_acc, val_loss, val_acc))
                else:
                    print("iteration: %d/%d, train_loss: %.4f, train_acc: %.4f" %(it, iterations, train_loss / it, train_acc / it) , end='\r')

        save_path = saver.save(sess, model_save_path)
        print("Model saved in file: %s" % save_path)        




          

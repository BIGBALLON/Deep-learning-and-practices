import keras
import math
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.initializers import RandomNormal  
from keras import optimizers
from keras.callbacks import LearningRateScheduler
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file

batch_size = 128
num_classes = 10
epochs = 164
iterations = 391

img_rows, img_cols = 32, 32
img_channels = 3
log_filepath = r'./logs/'
dropout = 0.5
weight_init = 0.0001

data_augmentation = True

def scheduler(epoch):
  learning_rate_init = 0.01
  if epoch >= 81:
    learning_rate_init = 0.001
  if epoch >= 122:
    learning_rate_init = 0.0001
  return learning_rate_init

def get_he_weight(k,c):
    return math.sqrt(2/(k*k*c))


# -------- vgg19-load weight path -------- #
WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
filepath = get_file('vgg19_weights_tf_dim_ordering_tf_kernels.h5', WEIGHTS_PATH, cache_subdir='models')


# -------- data loading -------- #

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# -------- data preprocessing -------- #

x_train[:,:,:,0] = (x_train[:,:,:,0]-123.680)
x_train[:,:,:,1] = (x_train[:,:,:,1]-116.779)
x_train[:,:,:,2] = (x_train[:,:,:,2]-103.939)
x_test[:,:,:,0] = (x_test[:,:,:,0]-123.680)
x_test[:,:,:,1] = (x_test[:,:,:,1]-116.779)
x_test[:,:,:,2] = (x_test[:,:,:,2]-103.939)


tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, histogram_freq=0)
change_lr = LearningRateScheduler(scheduler)
cbks = [change_lr,tb_cb]


# -------- build model -------- #
model = Sequential()
# Block 1
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,64)), name='block1_conv1', input_shape=x_train.shape[1:]))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,64)), name='block1_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

# Block 2
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,128)), name='block2_conv1'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,128)), name='block2_conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

# Block 3
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,256)), name='block3_conv1'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,256)), name='block3_conv2'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,256)), name='block3_conv3'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,256)), name='block3_conv4'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

# Block 4
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,512)), name='block4_conv1'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,512)), name='block4_conv2'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,512)), name='block4_conv3'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,512)), name='block4_conv4'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

# Block 5
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,512)), name='block5_conv1'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,512)), name='block5_conv2'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,512)), name='block5_conv3'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,512)), name='block5_conv4'))

# model modification for cifar-10
model.add(Flatten(name='flatten'))
model.add(Dense(4096, use_bias = True, kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = 0.01), activation='relu', name='fc_cifa10'))
model.add(Dropout(dropout))
model.add(Dense(4096, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = 0.01), name='fc2'))  
model.add(Dropout(dropout))      
model.add(Dense(10, activation='softmax', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = 0.01), name='predictions_cifa10'))        

# load pretrained weight from VGG19 by name      
model.load_weights(filepath, by_name=True)

# -------- optimizer setting -------- #
sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=cbks,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True,
            width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

    datagen.fit(x_train)

    model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                        steps_per_epoch=iterations,
                        epochs=epochs,
                        callbacks=cbks,
                        validation_data=(x_test, y_test))
    model.save('retrain.h5')

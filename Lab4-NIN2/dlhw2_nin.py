import keras
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

batch_size = 128
num_classes = 10
epochs = 164
data_augmentation = True

img_rows, img_cols = 32, 32
img_channels = 3
log_filepath = r'./logs/'
dropout = 0.5

def scheduler(epoch):
  learning_rate_init = 0.1
  if epoch >= 81:
    learning_rate_init = 0.01
  if epoch >= 122:
    learning_rate_init = 0.001
  return learning_rate_init


def build_model(dropout,i):

  global log_filepath
  op1 = op2 = op3 = False
  if i == 0:
    op1 = False
    op2 = False
    op3 = False
    log_filepath = r'./logs/none/'
  elif i == 1:
    op1 = False
    op2 = False
    op3 = True
    log_filepath = r'./logs/WI/'
  elif i == 2:
    op1 = False
    op2 = True
    op3 = False
    log_filepath = r'./logs/BN/'
  elif i == 3:
    op1 = False
    op2 = True
    op3 = True
    log_filepath = r'./logs/BN+WI/'
  elif i == 4:
    op1 = True
    op2 = False
    op3 = False
    log_filepath = r'./logs/ELU/'
  elif i == 5:
    op1 = True
    op2 = False
    op3 = True
    log_filepath = r'./logs/ELU+WI/'
  elif i == 6:
    op1 = True
    op2 = True
    op3 = False
    log_filepath = r'./logs/ELU+BN/'
  elif i == 7:
    op1 = True
    op2 = True
    op3 = True
    log_filepath = r'./logs/ELU+BN+WI/'
  print(log_filepath)

  acti_fun = 'relu'
  if op1 == True:
    acti_fun = 'elu'
  std_val1 = 0.01
  std_val2 = 0.05
  std_val3 = 0.05
  if op3 == True:
    std_val1 =  0.0204
    std_val2 =  0.0204
    std_val3 =  0.034



  model = Sequential()
  
  model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001), kernel_initializer=RandomNormal(stddev = std_val1), input_shape=x_train.shape[1:]))
  if op2 == True:
    model.add(BatchNormalization())  
  model.add(Activation(acti_fun))

  model.add(Conv2D(160, (1, 1)))
  if op2 == True:
    model.add(BatchNormalization())
  model.add(Activation(acti_fun))

  model.add(Conv2D(96, (1, 1)))
  if op2 == True:
    model.add(BatchNormalization())  
  model.add(Activation(acti_fun))

  model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same'))
  
  model.add(Dropout(dropout))

  model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001), kernel_initializer=RandomNormal(stddev = std_val2)))
  if op2 == True:
    model.add(BatchNormalization())
  model.add(Activation(acti_fun))

  model.add(Conv2D(192, (1, 1)))
  if op2 == True:
    model.add(BatchNormalization()) 
  model.add(Activation(acti_fun))

  model.add(Conv2D(192, (1, 1)))
  if op2 == True:
    model.add(BatchNormalization())  
  model.add(Activation(acti_fun))

  model.add(AveragePooling2D(pool_size=(3, 3),strides=(2,2),padding = 'same'))
  
  model.add(Dropout(dropout))
  model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.0001), kernel_initializer=RandomNormal(stddev = std_val3) ))
  if op2 == True:
    model.add(BatchNormalization())  
  model.add(Activation(acti_fun))

  model.add(Conv2D(192, (1, 1)))
  if op2 == True:
    model.add(BatchNormalization())  
  model.add(Activation(acti_fun))

  model.add(Conv2D(10, (1, 1)))
  if op2 == True:
    model.add(BatchNormalization())  
  model.add(Activation(acti_fun))
  
  #model.add(AveragePooling2D(pool_size=(8, 8),strides=(1,1)))
  model.add(GlobalAveragePooling2D())
  
  #model.add(Flatten())
  model.add(Activation('softmax'))

  sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
  model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
  return model

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

color_r = np.sum(x_train[:,:,:,0])/(50000.0*32*32)
color_g = np.sum(x_train[:,:,:,1])/(50000.0*32*32)
color_b = np.sum(x_train[:,:,:,2])/(50000.0*32*32)
variance_r = np.sqrt(np.sum(np.square(x_train[:,:,:,0] - color_r))/(50000.0*32*32-1)) 
variance_g = np.sqrt(np.sum(np.square(x_train[:,:,:,1] - color_g))/(50000.0*32*32-1)) 
variance_b = np.sqrt(np.sum(np.square(x_train[:,:,:,2] - color_b))/(50000.0*32*32-1)) 
x_train[:,:,:,0] = (x_train[:,:,:,0] - color_r) / variance_r
x_train[:,:,:,1] = (x_train[:,:,:,1] - color_g) / variance_g
x_train[:,:,:,2] = (x_train[:,:,:,2] - color_b) / variance_b

color_r = np.sum(x_test[:,:,:,0])/(10000.0*32*32)
color_g = np.sum(x_test[:,:,:,1])/(10000.0*32*32)
color_b = np.sum(x_test[:,:,:,2])/(10000.0*32*32)
variance_r = np.sqrt(np.sum(np.square(x_test[:,:,:,0] - color_r))/(10000.0*32*32-1))  
variance_g = np.sqrt(np.sum(np.square(x_test[:,:,:,1] - color_g))/(10000.0*32*32-1))  
variance_b = np.sqrt(np.sum(np.square(x_test[:,:,:,2] - color_b))/(10000.0*32*32-1))   
x_test[:,:,:,0] = (x_test[:,:,:,0] - color_r) / variance_r
x_test[:,:,:,1] = (x_test[:,:,:,1] - color_g) / variance_g
x_test[:,:,:,2] = (x_test[:,:,:,2] - color_b) / variance_b


for i in [6,7]:
  model = build_model(dropout,i)
  tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, histogram_freq=0)
  change_lr = LearningRateScheduler(scheduler)
  cbks = [change_lr,tb_cb]
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

      model.fit_generator(datagen.flow(x_train, y_train,
                                       batch_size=batch_size),
                          steps_per_epoch=391,
                          epochs=epochs,
                          callbacks=cbks,
                          validation_data=(x_test, y_test))
      #model.save('data_augmentation.h5')

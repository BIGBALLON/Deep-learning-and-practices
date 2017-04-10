## PART 1: Introduction
---

In this lab, we will use the pre-trained CNN model (VGG-19) to build an object recognition system. Second, we will retrain the pre-trained CNN model (VGG-19) on Cifar-10 dataset.

I used Keras to implement these models and test the following requirements:

- Object recognition system 
- Random initialization models
    - pure random initialization 
    - random initialization + WI
- Retrained models
    - pure retrain
    - retrain + WI 
    - retrain + WI + WD (0.0001,0.0005,0.001,0.0013,0.0015)
    - retrain + WI + BN + WD (0.0001,0.0005,0.001,0.0013,0.0015)

Lab Description: 

- VGG models
The main contribution of VGG model is a thorough evaluation of networks of increasing depth using an architecture with very small (𝟑 × 𝟑) convolution filters, which shows that a significant improvement on the prior-art configurations can be achieved by pushing the depth to 16–19 weight layers

- Object recognition system
    - VGG-19 model contains 1000 classes
    - Report the Top-5 classes

- Retrain VGG-19 on Cifar-10
    - Original input size is 224×224, Cifar-10 is 32×32
        - Change some layers
    - The output classes of VGG-19 are 1000, but Cifar-10 are 10 classes.
        - Change the final fully connected layer

## PART 2: Experiment setup
---


- Architecture Details

![lab51][1]

- weight file:  [vgg19_weights_tf_dim_ordering_tf_kernels.h5][2]

- Object recognition system:
    - Images are resized such that the smallest dimension becomes 224
    - then the center 224 × 224 crop is used

- Training hyperparameters:
    - Method: SGD with Nesterov momentum momentum=0.9, nesterov=True 
    - Data preprocessing: subtract RGB mean value 
        - B = 103.939 
        - G = 116.779
        - R = 123.680
    - Weight initialization: 
        - He’s WI
        - Random initialization
            - conv layer init = random_normal(stddev = 0.03)
            - fc layer init = random_normal(stddev = 0.01)
    - batch size: 128 (391 iterations for each epoch)
    - Total epochs: 164
    - Loss function: cross-entropy
    - Initial learning rate: 0.01, divide by 10 at 81, 122 epoch
    - Weight Decay: kernel_regularizer=keras.regularizers.l2(0.0001) also using 0.0005, 0.0010, 0.0013, 0.0015
    - Dropout: 0.5 
    - Batch normalization: w/wo

- Data augmentation: 
    - Translation: Pad 4 zeros in each side and random cropping back to 32x32 size 
    - Horizontal flipping with probability 0.5


## PART 3: Result
---


> Object recognition system:

![lab52][3]

![lab53][4]

> Retrain On Cifar-10:

- Final Accuracy (test error)

![lab54][5]

- Training loss curve:

![lab55][6]

- Test loss curve

![lab56][7]


## PART 4: Other experiments
---


- RI and Retrain w/wo WI

![lab57][8]

![lab58][9]

- Retrain + WI + WD

![lab59][10]

- Retrain + WI + WD + BN

![lab510][11]

## PART 5: Discussion 
---

- About overfitting:

According to Table 6 and Figure 7 & 8, overfitting will occur if the WD is too small (after echo 81, the accuracy of training data will reach 0.99 soon, overfitting occurred).
In other words, increase WD’s value can help us to avoid overfitting and get better results.
In Retrain+WI+BN+WD0.0015, we got the good test accuracy results 94.14%.

![lab511][12]


  [1]: http://7xi3e9.com1.z0.glb.clouddn.com/lab51.jpg
  [2]: https://github.com/fchollet/deep-learning-models/releases/
  [3]: http://7xi3e9.com1.z0.glb.clouddn.com/lab52.png
  [4]: http://7xi3e9.com1.z0.glb.clouddn.com/lab53.jpg
  [5]: http://7xi3e9.com1.z0.glb.clouddn.com/lab54.jpg
  [6]: http://7xi3e9.com1.z0.glb.clouddn.com/lab55.jpg
  [7]: http://7xi3e9.com1.z0.glb.clouddn.com/lab56.jpg
  [8]: http://7xi3e9.com1.z0.glb.clouddn.com/lab57.jpg
  [9]: http://7xi3e9.com1.z0.glb.clouddn.com/lab58.jpg
  [10]: http://7xi3e9.com1.z0.glb.clouddn.com/lab59.jpg
  [11]: http://7xi3e9.com1.z0.glb.clouddn.com/lab510.jpg
  [12]: http://7xi3e9.com1.z0.glb.clouddn.com/lab511.jpg
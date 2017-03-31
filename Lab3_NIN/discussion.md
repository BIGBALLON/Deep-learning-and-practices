
## PART 1: Introduction
---

As we know, traditional convolution neural network looks like 3x3 conv + ReLU, but in this Lab, we will try to build NIN (Network in Network) network architecture.   
(NIN: 3x3 conv + ReLU + 1x1 conv + ReLU + 1x1 conv +ReLU)

![][1]

We will train it on Cifar-10 dataset, the CIFAR-10 dataset consists of 60000 32×32 color images (RGB) in 10 classes,   
with 6000 images per class. There are 50000 training images and 10000 test images. 

![][2]

I using the Keras (A higher-level API based on Tensorflow) to implement the model.   
And I just use the following combination to train the model. (Assume Data preprocessing called DP, Weight Decay called WD, Weight initialization called WI, Nesterov momentum called NM, Data augmentation called DA)


- Model 1.  NIN + DA + DP + WD + WI + NM + Dropout0.5
- Model 2.  Model 1 without data augmentation 
- Model 3.  Model 1 without data preprocessing 
- Model 4.  Model 1 without data preprocessing & data augmentation
- Model 5.  New All convolution model


## PART 2: Experiment setup


> Model 1: NIN + DA + DP + WD + WI + NM + Dropout0.5

- Architecture Details:  
    - ConvPool-CNN is the same as TA’s architecture. 
    - And according to [STRIVING FOR SIMPLICITY: THE ALL CONVOLUTIONAL NET][3], I changed the ConvPool-CNN  model to the All-CNN architecture for Model 5.


![][4]

- Training hyperparameters:
    - Method: SGD with Nesterov momentum 
    - Data preprocessing: Color normalization
    - Weight initialization: stddev = 0.01 (for first conv layer)  0.05 (for others)
    - batch size: 128 (391 iterations for each epoch)
    - Total epochs: 164
    - Loss function: cross-entropy
    - Initial learning rate: 0.1, divide by 10 at 81, 122 epoch
    - Weight Decay: kernel_regularizer=keras.regularizers.l2(0.0001)
    - Dropout: 0.5 

- Data augmentation: 
    - Translation: Pad 4 zeros in each side and random cropping back to 32x32 size 
    - Horizontal flipping with probability 0.5

> Model 2: use Model 1 without Data augmentation  
> Model 3: use Model 1 without Color normalization  
> Model 4: use Model 1 without Data augmentation & Color normalization  

> Model 5: All convolution for Model 1 and Model 3


## PART 3: Result

We comparison the Model 1 and Model 2 (Model 1 without data augmentation)

- Final Accuracy (test error)

| Model 1 | Model 1 without data augmentation | 
| ------| ------ |
| 90.49%(9.51%) | 89.14%(10.86%) |

![][5]

- Training loss curve:

![][6]

- Test loss curve

![][7]

We got the final accuracy 90.49% with Data augmentation and 89.14% without Data augmentation.   
It is obvious that Data augmentation is important for training. According to the Training loss curve and Test error curve, we can learn that Data augmentation can help reduce the manual intervention required to developed meaningful information and insight of business data, as well as significantly enhance data quality.


## PART 4: Other experiments & Discussion

- Color normalization Test

I trained other three model based on Model 1, and we can see that the Color normalization trick provides about 4~5% improvement, it’s cool.

| Without data preprocessing: | Without data preprocessing | 
| ------| ------ |
|  With DA: 86.93%(Without DA: 84.30%) | With DA: 90.49%(Without DA:  89.14%) |

![][8]


- SGD with/without momentum Test

We can see that SGD with momentum (86.93%) is better than without momentum (86.25%)

![][9]

- The effect of Dropout rate

![][10]

![][11]

Dropout is a regularization technique for reducing overfitting in neural networks by preventing complex co-adaptations on training data. It is a very efficient way of performing model averaging with neural networks. From the Figure 5 and Figure 6, we can realize the value of dropout between 0.3 and 0.6 have a good performance.   
However, if the value is too large, the model may become overfitting, in Model 3, if dropout’s value is more than 0.7, training will be failed. 


- The All-CNN

I used the All-CNN model which I mentioned earlier. And I got lots of trouble at the time of training, sometimes the model’s accuracy will go up, sometimes will go down, finally, the accuracy will become 0.1 and training will be failed (just like orange line). I changed the initial learning rate from 0.1 to 0.05, but training still failed. Then, I used batch normalization to make it train fast, after 200 epochs’ training (about 6 hours), I get the blue line. The accuracy of this model is 91.33%.  
While in Model 3(Model 1 without Color normalization), it seems that New All-CNN’s performance is no better than ConvPool-CNN. I’m not sure, maybe there are some bug in my code?

![][12]

![][13]

All in all, it’s fantastic to learn deep learning and it’s very interesting.   
I learned from the assignment that each parameter in CNN is important, how to design the architecture, how to initialize the weight, the learning rate and which activation to choose, all this problem is important, I still have a long way to go!


  [1]: http://7xi3e9.com1.z0.glb.clouddn.com/Picture1.png
  [2]: http://7xi3e9.com1.z0.glb.clouddn.com/Picture2.png
  [3]: https://arxiv.org/pdf/1412.6806.pdf
  [4]: http://7xi3e9.com1.z0.glb.clouddn.com/ALLCNN.png
  [5]: http://7xi3e9.com1.z0.glb.clouddn.com/Picture3.png
  [6]: http://7xi3e9.com1.z0.glb.clouddn.com/Picture4.png
  [7]: http://7xi3e9.com1.z0.glb.clouddn.com/Picture5.png
  [8]: http://7xi3e9.com1.z0.glb.clouddn.com/Picture6.png
  [9]: http://7xi3e9.com1.z0.glb.clouddn.com/Picture7.png
  [10]: http://7xi3e9.com1.z0.glb.clouddn.com/Picture8.png
  [11]: http://7xi3e9.com1.z0.glb.clouddn.com/Picture9.png
  [12]: http://7xi3e9.com1.z0.glb.clouddn.com/Picture10.png
  [13]: http://7xi3e9.com1.z0.glb.clouddn.com/Picture11.png

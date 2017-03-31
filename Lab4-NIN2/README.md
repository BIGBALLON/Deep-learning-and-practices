
## PART 1: Introduction
---

In Lab 3, we learned how to build an NIN (Network in Network), and how to use Tensorflow (Keras or tflearn) to implement the model. In this Lab, we will be asked to use various activation functions, batch normalization, an weight initialization in NIN, and train it on Cifar-10 dataset.


I used Keras to implement these models and test the following requirements:

- Use ReLU, Leaky Rectifier Linear Unit(LReLU) , Exponential Linear Unit(ELU)  and Maxout activation function
- implement weight initialization method of He’s paper 
- use batch normalization in NIN model.
- Train NIN using three different activation functions and compare
    - 2(ReLU, ELU,) X 2(w/wo BN) X 2(w/wo weight initial)
    - 2(LReLU, Maxout) X 1(with BN) X 1(with weight initial)

> Lab Description

- Dying ReLUs

![][1]

- Leaky Rectifier Linear Unit

![][2]

- Exponential Linear Unit

![][3]

alpha = 1 in the experiments of the original paper

- He’s Weight initialization

![][4]

- Batch Normalization

![][5]

![][6]

## PART 2: Experiment setup

- Architecture Details

![AD][7]

- Training hyperparameters:
    - Method: SGD with Nesterov momentum momentum=0.9, nesterov=True 
    - Data preprocessing: Color normalization
    - Weight initialization: He's WI
    - batch size: 128 (391 iterations for each epoch)
    - Total epochs: 164
    - Loss function: cross-entropy
    - Initial learning rate: 0.1, divide by 10 at 81, 122 epoch
    - Weight Decay: kernel_regularizer=keras.regularizers.l2(0.0001)
    - Dropout: 0.5 

- Data augmentation: 
    - Translation: Pad 4 zeros in each side and random cropping back to 32x32 size 
    - Horizontal flipping with probability 0.5

## PART 3: Result

I just test all of the 8 models, it cost about 3 hours(2h59m) to train a model (ELU+BN+WI), (ELU+BN), (BN) and (BN+WI), 1h35min for the other models.  
All of the 8 CNN models' final test accuracy are more than 90%.   
The ELU+BN+WI model gets 91.67%, and the ELU+BN model gets 91.34%, it seems nice.   
We can learn that as the data flows through a deep network, the weights and parameters adjust those values, sometimes making the data too big or too small again. By normalizing the data in each mini-batch, this problem is largely avoided.  
See the table and figures below for detailed results

|Combination|Accuracy|
|---|---|
|ELU+BN+WI|91.67%|
|ELU+BN|91.34%|
|BN	|91.13%|
|BN+WI|90.98%|
|ELU+WI|90.94%|
|ELU|90.76%|
|None|90.50%|
|WI|90.49%|


![][8]

![][9]

However, I just used He's weight in three layers. After using it in all convolution layers, 

```
model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(op3,1,5,192)), input_shape=x_train.shape[1:]))
model.add(Conv2D(160, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(op3,2,1,160))))
model.add(Conv2D(96, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(op3,3,1,96)) ))
model.add(Conv2D(192, (5, 5), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(op3,4,5,192))))
model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(op3,5,1,192))))
model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(op3,6,1,192))))
model.add(Conv2D(192, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(op3,7,3,192))))
model.add(Conv2D(192, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(op3,8,1,192))))
model.add(Conv2D(10, (1, 1), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(op3,8,1,192))))
```

I got the good result:

|Combination|Accuracy|
|---|---|
|ELU+BN+WI|92.66%|
|ELU+BN|92.44%|
|BN+WI|92.21%|
|BN|91.90%|
|ELU|91.59%|
|None|90.97%|
|ELU+WI|10.00%|
|WI|10.00%|

![][10]

## PART 4: Other experiments & Discussion

In lab4, we used lots of tricks and method to improve our model’s accuracy.

BN (Batch normalization) faster learning and higher overall accuracy. The improved method also allows us to use a higher learning rate, potentially providing another boost in speed. (Initial learning rate 0.1 is too large to train if we did not use special tricks in Lab3, but BN can avoid this problem, it makes loss drops very fast.)

WI (ReLU + He's weight initial + without BN) got the worst accuracy compared with other models. And sometimes WI may not converge.

Different activation function will have different effect, ELU and LeakyReLU have better performance than ReLU. As for maxout, I tried to use it, but it seems that my code doesn't work. 

## PART 5: References

- Maas, A. L., Hannun, A. Y., & Ng, A. Y. (2013, June). [Rectifier nonlinearities improve neural network acoustic models][11]. In Proc. ICML (Vol. 30, No. 1).
- Clevert, D. A., Unterthiner, T., & Hochreiter, S. (2015). [Fast and accurate deep network learning by exponential linear units (elus)][12]. arXiv preprint arXiv:1511.07289.
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). [Delving deep into rectifiers: Surpassing human-level performance on imagenet classification][13]. In Proceedings of the IEEE International Conference on Computer Vision (pp. 1026-1034).
- Ioffe, S., & Szegedy, C. (2015). [Batch normalization: Accelerating deep network training by reducing internal covariate shift][14]. arXiv preprint arXiv:1502.03167.
- Goodfellow, I. J., Warde-Farley, D., Mirza, M., Courville, A. C., & Bengio, Y. (2013). [Maxout networks][15]. ICML (3), 28, 1319-1327.


  [1]: http://7xi3e9.com1.z0.glb.clouddn.com/Lab44.png
  [2]: http://7xi3e9.com1.z0.glb.clouddn.com/Lab45.png
  [3]: http://7xi3e9.com1.z0.glb.clouddn.com/Lab46.png
  [4]: http://7xi3e9.com1.z0.glb.clouddn.com/Lab49.png
  [5]: http://7xi3e9.com1.z0.glb.clouddn.com/Lab47.png
  [6]: http://7xi3e9.com1.z0.glb.clouddn.com/Lab48.png
  [7]: http://7xi3e9.com1.z0.glb.clouddn.com/AC.png
  [8]: http://7xi3e9.com1.z0.glb.clouddn.com/lab42.png
  [9]: http://7xi3e9.com1.z0.glb.clouddn.com/lab41.png
  [10]: http://7xi3e9.com1.z0.glb.clouddn.com/Lab43.png
  [11]: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.693.1422&rep=rep1&type=pdf
  [12]: https://arxiv.org/pdf/1511.07289.pdf
  [13]: https://arxiv.org/pdf/1502.01852.pdf
  [14]: https://arxiv.org/pdf/1502.03167.pdf
  [15]: http://jmlr.org/proceedings/papers/v28/goodfellow13.pdf
## PART 1: Introduction

In this project, we are going to build a LSTM structure to do the copy experiment.

I used Tensorflow to implement sequence-to-sequence models to solve the copy task. 
It has reasonable accuracy rate when training length >= testing length.
And also reasonable when training length < testing length (train= 20 + 10 padding, test = 30), test sequence = 50 cost lots of iteration to train and got an undesirable result.

I tested many optimizers(Adam, SGD, Momentum, RMSProp etc. ) 
And I also tested some RNN cells (BasicLSTMCell, GRUCell, MultiRNNCell etc.) 


## PART 2: Experiment setup
---


- Training hyperparameters:
    - Optimizer:
        - GradientDescentOptimizer(lr)
        - MomentumOptimizer(lr, 0.9, use_nesterov=True) 
        - AdamOptimizer(lr)
        - RMSPropOptimizer(learning_rate=0.0006, momentum=0.9)
    - batch size: 128 (also test 64 and 256)
    - Iteration: 10000/15000/100000
    - Sequence length: 10/20/30/50
    - Hidden size: 500
    - Embedding size: 100



## PART 3: Result
---

![][1]

![][2]

![][3]

## PART 4: Other experiments
---

![][4]

![][5]

## PART 5: Discussion 
---

![][6]


  [1]: http://7xi3e9.com1.z0.glb.clouddn.com/lab61.png
  [2]: http://7xi3e9.com1.z0.glb.clouddn.com/lab62.png
  [3]: http://7xi3e9.com1.z0.glb.clouddn.com/lab63.png
  [4]: http://7xi3e9.com1.z0.glb.clouddn.com/Lab64.png
  [5]: http://7xi3e9.com1.z0.glb.clouddn.com/lab65.png
  [6]: http://7xi3e9.com1.z0.glb.clouddn.com/lab66.png
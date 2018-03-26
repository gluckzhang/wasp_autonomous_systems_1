# Neural Network Training with Tensorflow

The requirements of this assignment can be read [on this page](https://kth.instructure.com/courses/4962/assignments/21979).  

Our goal is using TensorFlow to explore neural network and design 3 models to do the classification tasks on CIFAR-10 dataset.  

Short introduction to the 3 models:  

- Model 1: only 1-layer network trained with cross-entropy loss, mainly testing different parameters’ influence on results, test accuracy is around **0.35**
- Model 2: enhancing Model 1, adding an additional hidden layer and using momentum optimizer, test accuracy **0.3927**
- Model 3: implementation of a convolutional network, test accuracy **0.558**
# CENG506 Deep Learning Assignments
This repository is created for the Deep Learning graduate-level course practicals and assignments. (2020-2021 SPRING)

## Practicals
Practicals belong to the [DL course of UNIGE/EPFL given by Professor François Fleuret](https://fleuret.org/dlc/#information)

### Practical I: [PDF](https://github.com/AydinAlptug/CENG506-Deep-Learning-Assignments/blob/master/Practicals/Documents%20of%20Professor%20Fleuret/dlc-practical-1.pdf)
This practical contains 4 questions and the aim is to get familiar with the tensor operations with PyTorch.

### Practical II: [PDF](https://github.com/AydinAlptug/CENG506-Deep-Learning-Assignments/blob/master/Practicals/Documents%20of%20Professor%20Fleuret/dlc-practical-2.pdf)
This practical contains 4 questions and the aim is to deal with a data-set, KNN, PCA dimension reduction along with continuing to practice tensor operations.

***Rest will be added when I am available***

## Assignments
Assignments belong to the DL course of IZTECH provided by Asst. Prof. Mustafa Ozuysal.

### Assignment I
Evaluating the performance of the following fully-connected neural network architectures on the full MNIST dataset:

[A](https://github.com/AydinAlptug/CENG506-Deep-Learning-Assignments/blob/master/Assignments/Assignment%20I/A.py) - 2-layer NN, 300 hidden units, mean square error <br />
[B](https://github.com/AydinAlptug/CENG506-Deep-Learning-Assignments/blob/master/Assignments/Assignment%20I/B.py) - 3-layer NN, 300+100 HU, mean square error <br />
[C](https://github.com/AydinAlptug/CENG506-Deep-Learning-Assignments/blob/master/Assignments/Assignment%20I/C.py) - 3-layer NN, 500+300 HU, softmax, cross entropy, weight decay

Rules:

Do not use autograd or the higher level functionality (such as modules or optimizers) from torch. <br />
Use PyTorch only as a tensor library.

To see results, from terminal:

python mnist_classifier.py --arch=1 *(for 2-layer NN, 300 hidden units, mean square error)* <br />
python mnist_classifier.py --arch=2 *(for 3-layer NN, 300+100 hidden units, mean square error)* <br />
python mnist_classifier.py --arch=3 *(for 3-layer NN, 500+300 HU, softmax, cross entropy, weight decay)*

***Rest will be added when I am available***

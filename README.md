# Deep-Learning-for-Computer-Vision
Deep Learning for Computer Vision, COMP4901J @ HKUST.
This is actually an imitation course as CS231n @ Stanford. Therefore, the lecture notes and assignments are the same except the last assignment.

This repository may consist of four assignments for COMP4901J.

## Assignment 1
In this assignment you will practice putting together a simple image classification pipeline, based on the k-Nearest Neighbor or the SVM/Softmax classifier. The goals of this assignment are as follows:

- understand the basic Image Classification pipeline and the data-driven approach (train/predict stages)
- understand the train/val/test splits and the use of validation data for hyperparameter tuning.
- develop proficiency in writing efficient vectorized code with numpy
- implement and apply a k-Nearest Neighbor (kNN) classifier
- implement and apply a Multiclass Support Vector Machine (SVM) classifier
- implement and apply a Softmax classifier
- implement and apply a Two layer neural network classifier
- understand the differences and tradeoffs between these classifiers
- get a basic understanding of performance improvements from using higher-level representations than raw pixels (e.g. color histograms, Histogram of Gradient (HOG) features)

Download data: Once you have the starter code, you will need to download the CIFAR-10 dataset. Run the following from the assignment1 directory: cd /assignment1/cs231n/datasets folder and click get_datasets.py to run the Python code to download the data.

## Assignment 2
In this assignment you will practice writing backpropagation code, and training Neural Networks and Convolutional Neural Networks. The goals of this assignment are as follows:

- understand Neural Networks and how they are arranged in layered architectures
- understand and be able to implement (vectorized) backpropagation
- implement various update rules used to optimize Neural Networks
- implement batch normalization for training deep networks
- implement dropout to regularize networks
- effectively cross-validate and find the best hyperparameters for Neural Network architecture
- understand the architecture of Convolutional Neural Networks and train gain -  - experience with training these models on data

cd /assignment2/cs231n/datasets folder and click get_datasets.py to run the Python code to download the data.

## Assignment 3
n this assignment you will implement recurrent networks, and apply them to image captioning on Microsoft COCO. You will also explore methods for visualizing the features of a pretrained model on ImageNet, and also this model to implement Style Transfer. Finally, you will train a generative adversarial network to generate images that look like a training dataset!

The goals of this assignment are as follows:

- Understand the architecture of recurrent neural networks (RNNs) and how they operate on sequences by sharing weights over time
- Understand and implement both Vanilla RNNs and Long-Short Term Memory (LSTM) RNNs
- Understand how to sample from an RNN language model at test-time
- Understand how to combine convolutional neural nets and recurrent nets to implement an image captioning system
- Understand how a trained convolutional network can be used to compute gradients with respect to the input image
- Implement and different applications of image gradients, including saliency maps, fooling images, class visualizations.
- Understand and implement style transfer.
- Understand how to train and implement a generative adversarial network (GAN) to produce images that look like a dataset.

## Assignment 4
In this assignment you will practice writing simple reinforcement learning algorithms, and training DQN and Policy Gradient to solve the classical CartPole and WorldNavigate problems. The goals of this assignment are as follows:

- Practice how to use the OpenAI Gym interfaces
- Understand and be able to implement Q-table and Q-network learning algorithms
- Understand convolutional layers and how they can be applied in Q-learning and policy gradient methods
- Implement DQN to solve the world navigation task
- Implement two improved DQNs, the Double Q-learning and Duel DQN
- Build a policy-gradient based agent that can solve the CartPole task
- Combine model network and policy network without actually training on real environment

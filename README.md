# Neural-Network-Cpp
<img src="https://user-images.githubusercontent.com/40866831/195354581-f362cf20-3ca6-4dd4-891d-fb618ab4c56c.png" width="250"/>

Artificial Neural Network implemented in c++ language based on MNIST database

## Table of contents
* [General Info](#general-information)
* [Assumptions](#assumptions)
* [External specification - setup](#setup)
* [Internal specification](#internal-specification)
* [Construction specification](#construction-specification)
* [Files tree](#files-tree)
* [Code description](#code-description)
* [Neural network structure](#neural-network-structure)
* [How does it work?](#how-does-it-work)
* [Sources for acquiring knowledge](#sources-for-acquiring-knowledge)

## General Information 

A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way
the human brain operates.<br>
In this sense, neural networks refer to systems of
neurons, either organic or artificial in nature.<br>
Almost all commercial machine learning applications depend on artificial neural networks, which are trained using large datasets with a back-propagation
algorithm. <br>
The network first analyzes a training example, typically assigning
it to a classification bin. This result is compared to the known “correct” answer, and the difference between the two is used to adjust the weights applied to the network nodes.<br>
The process repeats for as many training examples as needed to (hopefully)
converge to a stable set of weights that gives acceptable accuracy. <br>
This standard algorithm requires two distinct computational paths — a forward
“inference” path to analyze the data, and a backward “gradient descent” path
to correct node weights

## Assumptions
The program gets the data from the files of MNIST database on the basis
of which it can perform the process of training the neural network - or it’s
testing.

## Setup 
My script implementation is command line program. For the puropse of preparing my solution, to build (compile) the program i used the ’make’ tool,
so it is incipiently included in the project.
Therefore, to generate the ’training’ and ’testing’ executable files you must
use the terminal command:

`$ make`

Generated executables do not require any additional arguments.
After generating them, in order to start the process of training the neural
network, you should run the:

`$ ./training`

Or in case of testing trained neural network:

`$ ./testing`

The program does not have any in-terminal short manual included - no such
option was needed.

## Internal Specification 

The program is not implemented with structural paradigm. <br>
Script does not contain a specific user interface, therefore such an application was not needed.<br>
Imperative paradigm was only used to separate the partially separated testing and training scripts.

##  Construction specification

The compiler used during creating the project: g++ (GCC) 12.1.0

language: c++

language standard: 17

## Files Tree

File <b>test.cpp</b> - code of testing a neural network process<br>
File <b>train.cpp</b> - code of training a neural network process<br>
File <b>test</b> - generated by compiling executable to perform testing process<br>
File <b>train</b> - generated by compiling executable to perform training process<br>
File <b>neural-network-model.dat</b> - file containing the weights of neural network<br>
Folder <b>/reports/:</b> - folder containing reports of program results<br>
File <b>/reports/testing-report.dat</b> - report file saving the results of testing<br>
File <b>/reports/training-report.dat</b> - report file saving the results of training<br>
Folder <b>/mnist/:</b> - folder containing MNIST database files

## Code description

Description of types and functions included in the target files in comments.

##  Neural network structure
<img src="https://user-images.githubusercontent.com/40866831/195360662-82c6a22e-8d37-41f6-856b-3747a41a20f8.jpg" alt="drawing" width="550"/>

## How does it work?

### Training:
The program begins its operation by opening and getting data from files:
report of any previous learning process and MNIST database - gray scale
image and the corresponding label.<br> The neural network is initialized starting
with the memory allocation for its layers. Then its weights are initialized.<br>
The program begins its neural network training based on backward propagation algorithm - calculates the gradient of the error function with respect to
the neural network’s weights.<br> Then, the error threshold for a given iteration
of the neural network training process is calculated.<br>In a loop, the program
displays the current image from the database, the number of iteration being
performed, the number of iterations of the training process for a given image
and the error threshold.<br> After the process is completed, the program saves
the current network - its weights in the model-neural-network.dat file.<br>
The program also saves a report on all individual iterations of the learning
process into training-report.dat

### Testing:
Identically, the program begins its operation by opening and getting data
from the database and the report.<br> The memory allocation is initialized and
the trained neural network model is read (weight matrices).<br> The network
goes through a classification process - perceptron procedure and the prediction.<br> The screen lists the classification resault and the error threshold.<br> Aftercompleting the number of iterations of testing indicated in the program, it
prints the number of correct samples compared to the number of iterations
performed and the overall accuracy.

## Sources for acquiring knowledge

### online
* [AI wiki](https://wiki.pathmind.com/neural-network)
* 3Blue1Brown[ But what is neural network?](https://www.youtube.com/watch?v=aircAruvnKk)
* 3Blue1Brown[ Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w)
* 3Blue1Brown[ What is backpropagation really doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
### books
* John Slavio[ Deep Learning and Artificial Intelligence: A Beginners’ Guide to Neural Networks and Deep Learning](https://www.amazon.com/Deep-Learning-Artificial-Intelligence-Beginners/dp/B07D4QZ6GC/ref=sr_1_2?crid=1XVOFGYWP3REL&keywords=Neural+Network%2C+John+Slavio&qid=1665583999&qu=eyJxc2MiOiIwLjQ5IiwicXNhIjoiMC4wMCIsInFzcCI6IjAuMDAifQ%3D%3D&sprefix=neural+network%2C+john+slavio%2Caps%2C265&sr=8-2)
* Jerzy Grębosz[ Opus Magnum c++](https://www.amazon.com/Opus-magnum-Misja-nadprzestrzen-17/dp/8328365871/ref=sr_1_1?crid=1MRC8Q55HCDBQ&keywords=opus+magnum+jerzy+gr%C4%99bosz&qid=1665584175&qu=eyJxc2MiOiIwLjc5IiwicXNhIjoiMC4wMCIsInFzcCI6IjAuMDAifQ%3D%3D&sprefix=opus+magnum+jerzy+gr%C4%99bos%2Caps%2C207&sr=8-1)

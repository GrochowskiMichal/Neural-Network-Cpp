/*
+-----------------------------------------------------------------------------------+
|   Project name: MNIST Artificial Neural Network Cpp                               |
|   Project version: 1.3                                                            |
|   File name: training.cpp                                                         |
|   Compilator: g++ (GCC) 12.1.0                                                    |
|   Project description: Artificial Neural Network working on MNIST's database.     |
|   Created by Michał Grochowski on 14 May 2022.                                    |
|                                                                                   |
|   Copyright © 2022 Michał Grochowski. All rights reserved.                        |
|                                                                                   |
|   Licensed under the Apache License, Version 2.0 (the "License");                 |
|   you may not use this file except in compliance with the License.                |
|   You may obtain a copy of the License at                                         |
|   http://www.apache.org/licenses/LICENSE-2.0                                      |
|   Unless required by applicable law or agreed to in writing, software             |
|   distributed under the License is distributed on an "AS IS" BASIS,               |
|   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.        |
|   See the License for the specific language governing permissions and             |
|   limitations under the License.                                                  |
+-----------------------------------------------------------------------------------+
*/

#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>

using namespace std;

// File containing training image data
const string training_data = "mnist/train-images.idx3-ubyte";

// File containing training image data labels
const string training_data_labels = "mnist/train-labels.idx1-ubyte";

// File containing weights of the artificial neural network
const string ann_model = "weights/neural-network-model.dat";

// File containing the report of training results
const string training_report = "reports/nn_training-report.dat";

// Number of training samples
const int noTraining = 60000;

// Size of the training data image - MNIST database
const int width = 28;   // width of common mnist database image
const int height = 28;  // height of common mnist database image

// number of input layer neurons 
const int n1 = width * height; // = 784, without bias neuron 

// Number of hidden layer neurons
const int n2 = 128;

// Number of output layer neurons
const int n3 = 10;

// Number of iterations for Back Propagation algorithm
const int epochs = 512;

// learning rate itself
const double learning_rate = 1e-3;

// Heuristic function optimizing Back Propagation algorithm
const double momentum = 0.9;

// Iterating process ends if learning error is smaller than epsilon
const double epsilon = 1e-3;

// From Input layer to Hidden layer
double *w1[n1 + 1], *delta1[n1 + 1], *out1;

// From Hidden layer to Output layer
double *w2[n2 + 1], *delta2[n2 + 1], *in2, *out2, *theta2;

// Output layer
double *in3, *out3, *theta3;
double expected[n3 + 1];

// Image in MNIST: 28x28 gray scale images.
int d[width + 1][height + 1];

// File stream to read data (image, label) and write down a report
ifstream image;
ifstream label;
ofstream report;

// Displaying information about the program

void info() {
	cout << "Training Artificial Neural Network for MNIST database" << endl;
	cout << endl;
	cout << "Number of input neurons: " << n1 << endl;
	cout << "Number of hidden neurons: " << n2 << endl;
	cout << "Number of output neurons: " << n3 << endl;
	cout << endl;
	cout << "Number of iterations: " << epochs << endl;
	cout << "Learning rate: " << learning_rate << endl;
	cout << "Momentum: " << momentum << endl;
	cout << "Epsilon: " << epsilon << endl;
	cout << endl;
	cout << "Training image data: " << training_data << endl;
	cout << "Training label data: " << training_data_labels << endl;
	cout << "Number of training sample: " << noTraining << endl << endl;
}

// Allocating the memory for artificial neural network

void init_array() {
	// From Input layer to Hidden layer
    for (int i = 1; i <= n1; ++i) {
        w1[i] = new double [n2 + 1];
        delta1[i] = new double [n2 + 1];
    }
    
    out1 = new double [n1 + 1];

	// From Hidden layer to Output layer
    for (int i = 1; i <= n2; ++i) {
        w2[i] = new double [n3 + 1];
        delta2[i] = new double [n3 + 1];
    }
    
    in2 = new double [n2 + 1];
    out2 = new double [n2 + 1];
    theta2 = new double [n2 + 1];

	// Output layer
    in3 = new double [n3 + 1];
    out3 = new double [n3 + 1];
    theta3 = new double [n3 + 1];
    
    // Initialization for weights from Input layer to Hidden layer
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
            int sign = rand() % 2;
            w1[i][j] = (double)(rand() % 6) / 10.0;
            if (sign == 1) {
				w1[i][j] = - w1[i][j];
			}
        }
	}
	
	// Initialization for weights from Hidden layer to Output layer
    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            int sign = rand() % 2;
            w2[i][j] = (double)(rand() % 10 + 1) / (10.0 * n3);
            if (sign == 1) {
				w2[i][j] = - w2[i][j];
			}
        }
	}
}

// Function of sigmoid
// A weighted sum of inputs is passed through an activation function and this output serves as an input to the next layer

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Forward process - Perceptron

void perceptron() {
    for (int i = 1; i <= n2; ++i) {
		in2[i] = 0.0;
	}

    for (int i = 1; i <= n3; ++i) {
		in3[i] = 0.0;
	}

    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
            in2[j] += out1[i] * w1[i][j];
		}
	}

    for (int i = 1; i <= n2; ++i) {
		out2[i] = sigmoid(in2[i]);
	}

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            in3[j] += out2[i] * w2[i][j];
		}
	}

    for (int i = 1; i <= n3; ++i) {
		out3[i] = sigmoid(in3[i]);
	}
}

// Error threshold

double square_error(){
    double res = 0.0;
    for (int i = 1; i <= n3; ++i) {
        res += (out3[i] - expected[i]) * (out3[i] - expected[i]);
	}
    res *= 0.5;
    return res;
}

// Back Propagation Algorithm

void back_propagation() {
    double sum;

    for (int i = 1; i <= n3; ++i) {
        theta3[i] = out3[i] * (1 - out3[i]) * (expected[i] - out3[i]);
	}

    for (int i = 1; i <= n2; ++i) {
        sum = 0.0;
        for (int j = 1; j <= n3; ++j) {
            sum += w2[i][j] * theta3[j];
		}
        theta2[i] = out2[i] * (1 - out2[i]) * sum;
    }

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            delta2[i][j] = (learning_rate * theta3[j] * out2[i]) + (momentum * delta2[i][j]);
            w2[i][j] += delta2[i][j];
        }
	}

    for (int i = 1; i <= n1; ++i) {
        for (int j = 1 ; j <= n2 ; j++ ) {
            delta1[i][j] = (learning_rate * theta2[j] * out1[i]) + (momentum * delta1[i][j]);
            w1[i][j] += delta1[i][j];
        }
	}
}

// Learning process: Perceptron, Back propagation

int learning_process() {
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
			delta1[i][j] = 0.0;
		}
	}

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
			delta2[i][j] = 0.0;
		}
	}

    for (int i = 1; i <= epochs; ++i) {
        perceptron();
        back_propagation();
        if (square_error() < epsilon) {
			return i;
		}
    }
    return epochs;
}

// Reading the input data - image and the corresponding to the image label 

void input() {
	// Reading data image
    char number;
    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            image.read(&number, sizeof(char));
            if (number == 0) {
				d[i][j] = 0; 
			} else {
				d[i][j] = 1;
			}
        }
	}
	
	cout << "Image:" << endl;
	for (int j = 1; j <= height; ++j) {
		for (int i = 1; i <= width; ++i) {
			cout << d[i][j];
		}
		cout << endl;
	}

    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            int pos = i + (j - 1) * width;
            out1[pos] = d[i][j];
        }
	}

	// Reading data image label
    label.read(&number, sizeof(char));
    for (int i = 1; i <= n3; ++i) {
		expected[i] = 0.0;
	}
    expected[number + 1] = 1.0;
    
    cout << "Label: " << (int)(number) << endl;
}

// Saving artificial neural network weights to .dat file

void write_matrix(string file_name) {
    ofstream file(file_name.c_str(), ios::out);
	
	// Input layer - Hidden layer
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
			file << w1[i][j] << " ";
		}
		file << endl;
    }
	
	// Hidden layer - Output layer
    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
			file << w2[i][j] << " ";
		}
        file << endl;
    }
	
	file.close();
}

// Main program function

int main(int argc, char *argv[]) {
	info();
	
    report.open(training_report.c_str(), ios::out);
    image.open(training_data.c_str(), ios::in | ios::binary); // Bin image data file
    label.open(training_data_labels.c_str(), ios::in | ios::binary ); // Bin image data label file

	// Reading file headers
    char number;
    for (int i = 1; i <= 16; ++i) {
        image.read(&number, sizeof(char));
	}
    for (int i = 1; i <= 8; ++i) {
        label.read(&number, sizeof(char));
	}
		
	// Artificial Neural Network Initialization

    init_array();
    
    for (int sample = 1; sample <= noTraining; ++sample) {
        cout << "Sample " << sample << endl;
        
        // Getting the data(image, label)
        input();
		
		// Learning process: Perceptron (Forward procedure) - Back propagation
        int noIterations = learning_process();

		// Write down the error threshold
		cout << "Number of iterations: " << noIterations << endl;
        printf("Error: %0.6lf\n\n", square_error());
        report << "Sample " << sample << ", Number of iterations = " << noIterations<< ", Error = " << square_error() << endl;
		
		// Save the current network (weights) into .dat file 
		if (sample % 100 == 0) {
			cout << "Saving Artificial neural network weights to " << ann_model << " file." << endl;
			write_matrix(ann_model);
		}
    }
	
	// Save the final artificial neural network (its weights) into .dat file 
    write_matrix(ann_model);

    report.close();
    image.close();
    label.close();
    
    return 0;
}

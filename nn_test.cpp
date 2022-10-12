/*
+-----------------------------------------------------------------------------------+
|   Project name: MNIST Artificial Neural Network Cpp                               |
|   Project version: 1.3                                                            |
|   File name: testing.cpp                                                          |
|   Compilator: g++ (GCC) 12.1.0                                                    |
|   Project description: Artificial Neural Network implementaion on MNIST's database|
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

// File containing testing image data
const string testing_data = "mnist/t10k-images.idx3-ubyte";

// File containing testing image data labels
const string testing_data_labels = "mnist/t10k-labels.idx1-ubyte";

// File containing weights of the artificial neural network
const string ann_model = "weights/model-neural-network.dat";

// File containing the report of testing results
const string testing_report = "reports/nn_testing-report.dat";

// Number of testing samples
const int nTesting = 10000;

// Size of the testing data image - MNIST database
const int width = 28;   // width of common mnist database image
const int height = 28;  // height of common mnist database image

// n1 = Number of input layer neurons
// n2 = Number of hidden layer neurons
// n3 = Number of output layer neurons

const int n1 = width * height; // = 784, without bias neuron 
const int n2 = 128; 
const int n3 = 10;

// From Input layer to Hidden layer
double *w1[n1 + 1], *out1;

// From Hidden layer to Output layer
double *w2[n2 + 1], *in2, *out2;

// Output layer
double *in3, *out3;
double expected[n3 + 1];

// Image In MNIST database: 28x28 gray scale images.
int d[width + 1][height + 1];

// Ifstream to read data (image, label) and write down testing report
ifstream image;
ifstream label;
ofstream report;

// Displaying informations about the program 

void info() {
	cout << "Testing Artificial Neural Network for MNIST database" << endl;
	cout << endl;
	cout << "Number of input neurons: " << n1 << endl;
	cout << "Number of hidden neurons: " << n2 << endl;
	cout << "Number of output neurons: " << n3 << endl;
	cout << endl;
	cout << "Testing image data: " << testing_data << endl;
	cout << "Testing image label data: " << testing_data_labels << endl;
	cout << "Number of testing sample: " << nTesting << endl << endl;
}

// Allocating the memory for artificial neural network

void init_array() {
	// From Input layer to Hidden layer
    for (int i = 1; i <= n1; ++i) {
        w1[i] = new double [n2 + 1];
    }
    
    out1 = new double [n1 + 1];

	// From Hidden layer to Output layer
    for (int i = 1; i <= n2; ++i) {
        w2[i] = new double [n3 + 1];
    }
    
    in2 = new double [n2 + 1];
    out2 = new double [n2 + 1];

	// Output layer
    in3 = new double [n3 + 1];
    out3 = new double [n3 + 1];
}

// Loading model of a trained Artificial Neural Network

void load_model(string file_name) {
	ifstream file(file_name.c_str(), ios::in);
	
	// Input layer - Hidden layer

    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
			file >> w1[i][j];
		}
    }
	
	// Hidden layer - Output layer

    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
			file >> w2[i][j];
		}
    }
	
	file.close();
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

// Reading the input data - image and the corresponding to the image label 

int input() {
	// Reading image
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
        
    return (int)(number);
}

// Main program function

int main(int argc, char *argv[]) {
	info();
    report.open(testing_report.c_str(), ios::out);
    image.open(testing_data.c_str(), ios::in | ios::binary); // Binary data image file
    label.open(testing_data_labels.c_str(), ios::in | ios::binary ); // Binary data image label file

	// Reading file headers

    char number;
    for (int i = 1; i <= 16; ++i) {
        image.read(&number, sizeof(char));
	}
    for (int i = 1; i <= 8; ++i) {
        label.read(&number, sizeof(char));
	}
		
	// Artificial Neural Network Initialization

    init_array(); // Memory allocation
    load_model(ann_model); // Loading model (weights matrix) of a trained Artificial Neural Network
    
    int correctSamples = 0;
    for (int sample = 1; sample <= nTesting; ++sample) {
        cout << "Sample " << sample << endl;
        
        // Getting the data(image, label)
        int label = input();
		
		// Classification - Perceptron procedure
        perceptron();
        
        // Prediction
        int prediction = 1;
        for (int i = 2; i <= n3; ++i) {
			if (out3[i] > out3[prediction]) {
				prediction = i;
			}
		}
		--prediction;

		// listing the classification result and the error threshold

		double error = square_error();
		printf("Error: %0.6lf\n", error);
		
		if (label == prediction) {
			++correctSamples;
			cout << "Classification: YES. Label = " << label << ". Prediction = " << prediction << endl << endl;
			report << "Sample " << sample << ": YES. Label = " << label << ". Prediction = " << prediction << ". Error = " << error << endl;
		} else {
			cout << "Classification: Number of Label = " << label << ". Prediction = " << prediction << endl;
			cout << "Image:" << endl;
			for (int j = 1; j <= height; ++j) {
				for (int i = 1; i <= width; ++i) {
					cout << d[i][j];
				}
				cout << endl;
			}
			cout << endl;
			report << "Sample " << sample << ": Number of Label = " << label << ". Prediction = " << prediction << ". Error = " << error << endl;
		}
    }

	// Summary
    double accuracy = (double)(correctSamples) / nTesting * 100.0;
    cout << "Number of correct samples: " << correctSamples << " / " << nTesting << endl;
    printf("Accuracy: %0.2lf\n", accuracy);
    
    report << "Number of correct samples: " << correctSamples << " / " << nTesting << endl;
    report << "Accuracy: " << accuracy << endl;

    report.close();
    image.close();
    label.close();
    
    return 0;
}

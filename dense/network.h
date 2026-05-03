#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <iostream>
#include <cstdlib>
#include <Eigen/Dense>

#define NUM_LAYERS 4

#define BATCH_SIZE 1024
#define STEP_SIZE 0.05 // LEARNING RATE
#define EPOCHS 100


// NOTE: I'll separate the header definitions and the implementations later on, this is not good practice. This is just a temp thing.


typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;



class Network
{
public:
	int* layers[NUM_LAYERS];
	
	Network(int layerCount, int* layerSizes);
	void forward(RowVector* layer);
	void backward(RowVector* targetValues);

private:
	float sigmoid(float x);
	float sigmoidDerivative(float x);
	void relu(RowVector* output);
	
	std::vector<RowVector*> neuronLayers;		// Store layers of neurons (vector of vectors) w/ each vec. having the act. number.
	std::vector<RowVector*> deltas;			// Error val vectors for layers
	std::vector<Matrix*> weights;	// Store the connections
	std::vector<Matrix*> gradient;	// Gradient to be applied during backprop. from error calculation
};

Network::Network(int layerCount, int* layerSizes) {

	
	for (int i = 0; i < layerCount; i++) {
		layers = layerSizes[i];
		// Add a new layer of neurons with a given size, and init at zero
		RowVector* newVec = RowVector::Zero(23);
		neuronLayers.push_back(newVec);
		deltas.push_back(new RowVector(neuronLayers.size()));
				
		// Add new matrix to keep track of weights for fully connected edges, but only between layers!
		if (i > 0) {
			// Initialize new matrix for weights and have each element be a random number.
			MatrixXf newWeightMatrix = MatrixXf::random(layerSizes[i - 1], layerSizes[i]);
			weights.push_back(newWeightMatrix);
			// Init the new weights with random values
		}
	}		
}

// Recall that the input layer will be images
void Network::forward(RowVector* inputLayer)
{

	// Grab layer length from array in pub.
	for (int i = 1; i < inputLayer.size(); i++) 
	{		

		if (i == inputLayer.Size() - 1) 
		{
			// Peform sigmoid on the output layer (last layer). This should hit the last layer also.
			(*inputLayer[i]) = sigmoid((*inputLayer[i - 1]) * (*weights[i - 1]));
		} 
		else
		{ 
			// Peform relu on the hidden layers
			(*inputLayer[i]) = (*inputLayer[i - 1]) * (*weights[i - 1]);
			relu(inputLayer);

		}
	}


}

// Use output vector as input!
void Network::relu(RowVector* input) 
{
	for (int i = 0; i < input.size() - 1; i++) {
		input[i] = std::max(0, input[i]);
	}
}

void Network::backward(RowVector* targetValues) {
	// Calculate the error and then update the weights going backwards. Recall that error is calculated on the last layer

	// Calculate error vector for ouput layer (last layer)
	deltas.back() = *targetValues - neuronLayers.back();

	// Go back through the hidden layers and update weights based on error vector values
	for (int i = neuronLayers.size() - 2; i > 0; i--) {
		(*deltas[i]) = (*deltas[i + 1]) * (*weights[i + 1].transpose());
	}

	// From there, compute the gradient going backwards
	/*
		Iterate over all neurons in each layer
		Inside this loop, update the the weight matrix corresponding to the layer (double nested loop over matrix)
	*/
	for (int i = 0; i < weights.size() - 1; i++) {

		gradient[i] = (neuronLayers[i].transpose() * deltas[i + 1]);

		for (int r = 0; r < weights[i]->rows(); r++) {
			for (int c = 0; c < weights[i]->cols(); c++) {

				float influence = sigmoidDerivative(gradient[i]->coeffRef(c));
				*weights->coeffRef(r, c) += (*deltas[i]->coeffRef(c)) * influence) * STEP_SIZE;

			}
		}
	}
}
	

// Use sigmoid for last layer
float Network::sigmoid(float x) {
	return 1 / (1 - expf(-x));
}

float Network::sigmoidDerivative(float x) {
	return (sigmoid(x) * (1 - sigmoid(x));
}

#endif
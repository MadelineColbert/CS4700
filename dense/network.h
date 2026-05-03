#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <iostream>
#include <cstdlib>
#include <eigen3/Eigen/Eigen>

#define NUM_LAYERS 4

#define BATCH_SIZE 1024
#define STEP_SIZE 0.05
#define EPOCHS 100


// NOTE: I'll separate the header definitions and the implementations later on, this is not good practice. This is just a temp thing.

typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;



class Network
{
public:
	int layers[NUM_LAYERS];
	
	Network(int layerCount, int* layerSizes);
	void forward(RowVector* layer);
	void backward(Network* network, float* inputLayer, float* outputLayer, RowVector* targetValues);
	void relu(RowVector* output);

private:
	double sigmoid(float x);
	
	std::vector<RowVector*> neuronLayers;		// Store layers of neurons (vector of vectors) w/ each vec. having the act. number.
	std::vector<RowVector*> deltas;			// Error val vectors for layers
	std::vector<Matrix*> weights;	// Store the connections
	std::vector<Matrix*> gradient;	// Gradient to be applied during backprop. from error calculation
};

Network::Network(int layerCount, int* layerSizes) {

	
	for (int i = 0; i < layerCount; i++) {
		layers = layerSizes[i];
		// Add a new layer of neurons with a given size, and init at zero
		neuronLayers.push_back(new RowVector::Zero(layerSizes[i]));
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

Network::forward(RowVector* inputLayer)
{
	// For fully connected neurons, get each val from the prev. layer..?
	float sum = 0.0f;

	// Grab layer length from array in pub.
	for (int i = 1; i < inputLayer.size(); i++) 
	{		
		(*inputLayer[i]) = sigmoid((*inputLayer[i - 1]) * (*weights[i - 1]));
	}

	relu(inputLayer);
}

// Use output vector as input!
Network::relu(RowVector* input) 
	{
	for (int i = 0; i < input.size() - 1; i++) {
		input[i] = std::max(0, input[i]);
	}
}

Network::backward(Network* network, float* inputLayer, float* outputLayer, RowVector* targetValues) {
	// Calculate the error and then update the weights going backwards. Recall that error is calculated on the last layer
	
	// Calculate error vector for ouput layer (last layer)
	(*deltas.back()) = targetValues - (*neuronLayers.back());
	
	// Go back through the hidden layers and update weights based one error vector values
	for (int i = layers[i] - 2; i > 0; i--) {
		(*delta[i]) = (*delta[i + 1]) * (*weights[i + 1]);
	}

	// From there, compute the gradient going backwards
}

Network::sigmoid(float x) {
	return 1 / (1 - expf(-x));
}


#endif
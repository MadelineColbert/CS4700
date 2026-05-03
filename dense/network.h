#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <iostream>
#include <cstdlib>
#include <eigen>

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
	
	Network(int layerCount);
	void forward(vector<float>* layer);
	void backward(Network* network, float* inputLayer, float* outputLayer);
	void softmax(RowVector* output);
	void relu(RowVector* output);
	void updateWeight()

private:
	
	std::vector<RowVector*> neuronLayers;		// Store layers of neurons (vector of vectors)
	std::vector<Matrix*> weights;	// Store the connections
};

Network::Network(int layerCount, int* layerSizes) {


	for (int i = 0; i < layerCount; i++) {
		// Add a new layer of neurons with a given size;
		neuronLayers.push_back(new RowVector(layerSizes[i]);

		// Add new matrix to keep track of weights for fully connected edges, but only between layers!
		if (i > 0) {
			weights.push_back(new Matrix(layerSize[i - 1] * layerSize[i]));
		}
	}

	// Randomization



}

Network::forward(vector<float>* inputLayer) {
	std::cout << "Yuh" << std::endl;

	// For fully connected neurons, get each val from the prev. layer..?
	float sum = 0.0f;

	// Grab layer length from array in pub.
	for (int i = 1; i < inputLayer.size(); i++) {
		inputLayer[i] = inputLayer[i - 1] * this->weights[i - 1];
		// Put relu func. ?
	}
}

Network::relu(RowVector* output) {

	for (int i = 0; i < output.size() - 1; i++) {
		output[i] = std::max(0, output[i]);
	}

}

Network::backward(Network* network, float* inputLayer, float* outputLayer) {


}

Network::softmax(RowVector* input) {

}



#endif
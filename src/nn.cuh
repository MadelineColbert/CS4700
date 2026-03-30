#ifndef NN_CUH__
#define NN_CUH__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
    float** weights;
    float* biases;
    int inputs;
    int outputs;
} Layer;

typedef struct{
    Layer* layers;
    int inputs;
    int outputs;
    int number_of_layers;
} NN;

NN* define_nn(int in, int out, int* layers, int n_layers);

void free_nn(NN* nn);

float inner_product(float * v1, float* v2, int n);

float* forward_pass(NN* nn, float* input);

#endif
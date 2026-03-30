#include "nn.cuh"

NN* define_nn(int in, int out, int* layers, int n_layers){
    int prev = in;
    NN* nn = (NN*)malloc(sizeof(NN));

    nn->inputs = in;
    nn->outputs = out;
    nn->number_of_layers = n_layers + 1;

    nn->layers = (Layer *)malloc(nn->number_of_layers * sizeof(Layer));

    printf("Number of Layers: %d\n", nn->number_of_layers);

    for (int i = 0; i <= n_layers; i++) {
        Layer* l = &nn->layers[i];
        l->inputs = prev;
        if (i == n_layers){
            l->outputs = out;
        }
        else{
            l->outputs = layers[i];
        }
        l->biases = (float*)calloc(l->outputs, sizeof(float));
        l->weights = (float **)calloc(l->outputs, sizeof(float*));
        for (int j =0; j<l->outputs; j++) {
            l->weights[j] = (float * )calloc(l->inputs, sizeof(float));
        }
        prev = layers[i];
    }

    return nn;
}

void free_nn(NN* nn){
    for (int i = 0; i < nn->number_of_layers; i++){
        Layer* l = &nn->layers[i];
        free(l->biases);
        for (int j =0; j<l->outputs; j++) {
            l->weights[j] = (float * )calloc(l->inputs, sizeof(float));
        }
        free(l->weights);
    }
    free(nn->layers);
    free(nn);
}


float inner_product(float * v1, float* v2, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += v1[i] * v2[i];
    }
    return sum;
}

float* forward_pass(NN* nn, float* input) {
    for (int lyr =0; lyr < nn->number_of_layers; lyr++){
        Layer* current_layer = &nn->layers[lyr];
        float* output = (float*)calloc(current_layer->outputs, sizeof(float));
        for (int o=0; o < current_layer->outputs; o++) {
            output[o] = inner_product(current_layer->weights[o], input, nn->inputs) + current_layer->biases[0];
            //output[o] = ACTIVATION(output[o]) (For now it's just Sigmoid)
            output[o] = 1/(1 + exp(-1 * output[o]));
        }
        input = output;
    }
    // This is output at the end
    return input;
}

void back_prop(NN* nn, float* input, float* output, float lr) {
    float* prediction = forward_pass(nn, input);
    float error = 0.0f;
    for (int i =0; i< nn->outputs; i++) {
        error += .5 * powf(prediction[i] - output[i], 2);
    }
}
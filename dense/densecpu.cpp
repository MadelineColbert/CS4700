#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include "network.h"

#define IMAGE_NEURONS 784
#define LAYER_1 100
#define LAYER_2 100
#define LAYER_3 100
#define RESULT_LAYER 10


double sigmoid(double x) {
     return 1.0 / (1 + std::exp(-x));
}

/* Inspiration on how to load (especially the magic numbers)
*  Was taken from https://github.com/projectgalateia/mnist/blob/master/mnist.h
*/
void load_mnist(char* image_file, char* label_file, float** images, int** labels, int* count) {
    FILE* ifp = fopen(image_file, "rb");
    FILE* lfp = fopen(label_file, "rb");
    char tmp[4];

    fread(tmp, 1, 4, ifp);
    if (mnist_bin_to_int(tmp) != 2051) {
        printf("Error");
    }

    fread(tmp, 1, 4, lfp);
    if (mnist_bin_to_int(tmp) != 2049) {
        printf("Error");
    }

    fread(tmp, 1, 4, ifp);
    int image_cnt = mnist_bin_to_int(tmp);

    fread(tmp, 1, 4, lfp);
    int label_cnt = mnist_bin_to_int(tmp);

    fread(tmp, 1, 4, ifp);
    int dim_1 = mnist_bin_to_int(tmp);

    fread(tmp, 1, 4, ifp);
    int dim_2 = mnist_bin_to_int(tmp);

    *count = image_cnt;
    *images = (float*)malloc(sizeof(float) * image_cnt * 28 * 28);
    *labels = (int*)malloc(sizeof(int) * image_cnt);

    for (int i = 0; i < image_cnt; i++) {
        unsigned char read_data[28 * 28];
        fread(read_data, 1, 28 * 28, ifp);
        for (int j = 0; j < 28 * 28; j++) { 
            (*images)[i * 28 * 28 + j] = read_data[j] / 255.0f;
        }
        fread(tmp, 1, 1, lfp);
        (*labels)[i] = (int)tmp[0];
    }


    fclose(ifp);
    fclose(lfp);
}

void free_input(float* images, int* labels, int count) {
    free(images);
    free(labels);
}

int main() {
    Network neuralNet;



    float* images = NULL;
    int* labels = NULL;
    int count = 0;

    load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte", &images, &labels, &count);
}
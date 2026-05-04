#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "network.h"
#include <chrono>
#include <time.h>

#define IMAGE_NEURONS 784
#define LAYER_1 100
#define LAYER_2 100
#define LAYER_3 100
#define RESULT_LAYER 10


// This + free_input() are from the .cu
int mnist_bin_to_int(char* v) {
    int i;
    int ret = 0;
    for (i = 0; i < 4; i++) {
        ret <<= 8;
        ret |= (unsigned char)v[i];
    }
    return ret;
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

// May not need this...?
float lossCompute(int outputLayerDim, int* labels) {
    return 1.0;
}

int main() {
    int layerConfig[] = {IMAGE_NEURONS, LAYER_1, LAYER_2, LAYER_3, RESULT_LAYER};

    // Init. the network
    Network* neuralNet = &Network::Network(5, layerConfig);

    // Prepare the dataset
    float* images = NULL;
    int* labels = NULL;
    int count = 0;

    int batches_per_epoch = BATCH_SIZE / EPOCHS;

    // Load images into 
    load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte", &images, &labels, &count);


    // Load image array into a row vector type
    RowVector imageInput = Eigen::Map<RowVector>(images, count);
    


    auto start = std::chrono::steady_clock::now();
    // Train the neural network
    for (int iter = 0; iter < EPOCHS; iter++) {
        for (int batch = 0; batch < batches_per_epoch; batch++) {
            // Since the batches per epoch is just integer division, we remove the remaining images.
            int offset = batch * BATCH_SIZE;

            // Call forward pass
            neuralNet->forward(&imageInput);

            // Call backward pass
            neuralNet->backward(&imageInput);

            // Compute the loss func
            float loss = lossCompute(RESULT_LAYER, labels);

            // Print accuracy...?
            printf("Epoch %d  batch %d/%d  loss = %f\n", iter + 1, batch, batches_per_epoch, loss);
            
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time taken: %f\n" << duration.count() << "ms" << std::endl;

    free(images);
    free(labels);

}
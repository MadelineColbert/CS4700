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

#define EPOCHS 100
#define BATCH_SIZE 1024

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
void load_mnist(const char* image_file, const char* label_file, float** images, int** labels, int* count) {
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

// TODO: Gonna need to work on 
float lossCompute(RowVector* results, RowVector targetVals) {
    return 1.0;
}

int main() {
    int layerConfig[] = {IMAGE_NEURONS, LAYER_1, LAYER_2, LAYER_3, RESULT_LAYER};

    // Init. the network
    Network* neuralNet = new Network(5, layerConfig);

    // Prepare the dataset
    float* images = NULL;
    int* labels = NULL;
    int count = 0;

    int batches_per_epoch = BATCH_SIZE / EPOCHS;

    // Load images into holders
    const char* imageTrain = "train-images.idx3-ubyte";
    const char* labelsTrain = "train-labels.idx1-ubyte";
    load_mnist(imageTrain, labelsTrain, &images, &labels, &count);



    auto start = std::chrono::steady_clock::now();
    // Train the neural network
    for (int iter = 0; iter < EPOCHS; iter++) {
        for (int batch = 0; batch < batches_per_epoch; batch++) {
            int offset = batch * BATCH_SIZE;
            
            RowVector inputVec = Eigen::Map<RowVector>(images + iter * 784, 784);
            // Create target vec for each iter
            int label = labels[iter];
            RowVector targetVec(10);
            targetVec.setZero();
            targetVec(label) = 1.0f;

            // Call forward pass
            neuralNet->forward(&inputVec);

            // Call backward pass
            neuralNet->backward(&targetVec);

            // Compute the loss func
            // TODO: Figure this part out, maybe? Not sure if completely necessary.
            RowVector* results = neuralNet->getResults(); 
            float loss = lossCompute(results, targetVec);

            // Print accuracy...?
            printf("Epoch %d  batch %d/%d  loss = %f\n", iter + 1, batch, batches_per_epoch, loss);
        }
    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time taken: " << duration.count() << " ms" << std::endl;

    free(images);
    free(labels);

    return 0;
}
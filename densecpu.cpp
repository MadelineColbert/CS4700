#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

// Maybe put into a header file?
#define NUM_LAYERS 1;


typdef struct {
    float* weights[NUM_LAYERS];
    float* bias[NUM_LAYERS];
    float* pre_act[NUM_LAYERS];
    float* post_act[NUM_LAYERS];
    float* post_grad[NUM_LAYERS];
    float* w_grad[NUM_LAYERS];
    float* b_grad[NUM_LAYERS];
    int layers[NUM_LAYERS + 1];
} NN_t;

// Recall le math func. and think about possible CPU optimizations
void softmax() {
    float* src = malloc()
}

void forward(NN* network, float* input){
    

}

void backward(NN* network, float* input, int* labels) {
   
}

void weightBiasInit(NN* network) {



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
    NN_t Network;
    network.layers[0] = 784;
    network.layers[1] = 100;
    network.layers[2] = 100;
    network.layers[3] = 100;
    network.layers[4] = 10;


}
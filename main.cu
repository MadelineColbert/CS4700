#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "cublas_v2.h"

/*
TODO:
- Implement better softmax kernel
- Update the NN struct to allow for gradients
- Implement backpropagation using cross-entropy loss
    - Biases
    - Weights
    - Handling activation functions (ReLU and Softmax)
- Add loss tracking over epochs
- Add sparsity measure
*/


#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t _e = (call);                                          \
        if (_e != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error %s:%d — %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(_e));          \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

#define CUBLAS_CHECK(call)                                                \
    do {                                                                  \
        cublasStatus_t _s = (call);                                       \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "cuBLAS error %s:%d — code %d\n",            \
                    __FILE__, __LINE__, (int)_s);                         \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

#define NUM_LAYERS 2

#define BATCH_SIZE 256
#define STEP_SIZE 0.01
#define EPOCHS 1

typedef struct{
    float* weights[NUM_LAYERS];
    float* bias[NUM_LAYERS];
    float *pre_act[NUM_LAYERS];
    float *post_act[NUM_LAYERS];
    int layers[NUM_LAYERS+1];
} NN;

__global__ void relu_kernel(float* pre_act,
                            float* post_act,
                            int out_dim,
                            int batch)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < out_dim && col < batch){
        int index = col*out_dim+row;
        post_act[index] = fmaxf(0.0f, pre_act[index]);
    }
}

//
__global__ void softmax_kernel(float* in, float* out, int length, int batch_size){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= batch_size) return;

    const float* src = in + col * length;
    float*       dst = in  + col * length;

    /* 1. find max for numerical stability */
    float mx = src[0];
    for (int i = 1; i < length; i++) mx = fmaxf(mx, src[i]);

    /* 2. exponentiate and sum */
    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        dst[i] = expf(src[i] - mx);
        sum   += dst[i];
    }

    /* 3. normalise */
    for (int i = 0; i < length; i++) dst[i] /= sum;
}

__global__ void add_bias(float* data, float* bias, int out_dim, int batch){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < out_dim && col < batch){
        int index = col*out_dim+row;
        data[index] += bias[row];
    }
}

void init_nn(NN* network) {
    for (int l = 0; l < NUM_LAYERS; l++) {
        int in_dim  = network->layers[l];
        int out_dim = network->layers[l + 1];

        float* h_W = (float*)calloc(in_dim * out_dim, sizeof(float));
        float* h_b = (float*)calloc(out_dim, sizeof(float));
        for (int i=0; i<in_dim*out_dim; i++){
          h_W[i] = 1;
        }
        CUDA_CHECK(cudaMemcpy(network->weights[l], h_W, in_dim * out_dim * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(network->bias[l], h_b, out_dim * sizeof(float),
                              cudaMemcpyHostToDevice));
        free(h_W);
        free(h_b);
    }
}


void define_nn(NN* network){
    for (int l=0; l< NUM_LAYERS; l++){
        int in_dim  = network->layers[l];
        int out_dim = network->layers[l + 1];
        CUDA_CHECK(cudaMalloc(&network->weights[l], in_dim*out_dim*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&network->bias[l], out_dim*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&network->pre_act[l], BATCH_SIZE*out_dim*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&network->post_act[l], BATCH_SIZE*out_dim*sizeof(float)));
    }
}

void free_nn(NN* network){
    for (int l=0; l< NUM_LAYERS; l++){
        CUDA_CHECK(cudaFree(network->weights[l]));
        CUDA_CHECK(cudaFree(network->bias[l]));
        CUDA_CHECK(cudaFree(network->pre_act[l]));
        CUDA_CHECK(cudaFree(network->post_act[l]));
    }
}

void forward(NN* network, float* input, cublasHandle_t handle){
    float one = 1.0f;
    float zero= 0.0f;
    float* in;
    for (int l=0; l<NUM_LAYERS; l++) {
        if (l == 0){
            in = input;
        } else{
            in = network->post_act[l-1];
        }
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        network->layers[l+1], BATCH_SIZE, network->layers[l],
                        &one, network->weights[l], network->layers[l+1],
                        in, network->layers[l], &zero,
                        network->pre_act[l], network->layers[l+1]));
        dim3 blockDim(16,16);
        dim3 gridDim((network->layers[l+1] + blockDim.x - 1) / blockDim.x,
                  (BATCH_SIZE + blockDim.y - 1) / blockDim.y);

        add_bias<<<gridDim, blockDim>>>(network->pre_act[l], network->bias[l], network->layers[l+1], BATCH_SIZE);

        if (l == NUM_LAYERS-1){
            softmax_kernel<<<gridDim, blockDim>>>(network->pre_act[l], network->post_act[l], network->layers[l+1], BATCH_SIZE);
        } else {
            relu_kernel<<<gridDim, blockDim>>>(network->pre_act[l], network->post_act[l],network->layers[l+1], BATCH_SIZE);
        }
    }
}

void backward(NN* network, float* input, int* labels, cublasHandle_t handle){
    float one = 1.0f;
    float zero= 0.0f;
    // Start by initializing with cross entropy loss


    // Average it, so gradient / BATCH_SIZE
    for (int l=NUM_LAYERS-1; l>=0; l--){

    }
}

int mnist_bin_to_int(char *v){
    int i;
    int ret = 0;
    for (i=0; i<4;i++){
        ret <<=8;
        ret |= (unsigned char)v[i];
    }
    return ret;
}

// Inspiration on how to load (especially the magic numbers)
// Was taken from https://github.com/projectgalateia/mnist/blob/master/mnist.h
void load_mnist(char* image_file, char* label_file, float** images, int** labels, int* count){
    FILE *ifp = fopen(image_file, "rb");
    FILE *lfp = fopen(label_file, "rb");
    char tmp[4];

    fread(tmp, 1, 4, ifp);
    if (mnist_bin_to_int(tmp) != 2051){
        printf("Error");
    }

    fread(tmp, 1, 4, lfp);
    if (mnist_bin_to_int(tmp) != 2049){
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
    *images = (float*)malloc(sizeof(float)*image_cnt*28*28);
    *labels = (int*)malloc(sizeof(int)*image_cnt);

    for (int i=0; i< image_cnt; i++) {
        unsigned char read_data[28*28];
        fread(read_data, 1, 28*28, ifp);
        for (int j=0; j < 28*28; j++){
            (*images)[i*28*28 + j] = read_data[j] / 255.0f;
        }
        fread(tmp, 1, 1, lfp);
        (*labels)[i] = (int)tmp[0];
    }

    fclose(ifp);
    fclose(lfp);
}

void free_input(float* images, int* labels, int count){
    free(images);
    free(labels);
}

int main() {
  NN network;
  cublasHandle_t handle;
  network.layers[0] = 784;
  network.layers[1] = 100;
  network.layers[2] = 10;

  float* images = NULL;
  int* labels=NULL;
  int count=0;

  load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte", &images, &labels, &count);

  define_nn(&network);

  init_nn(&network);

  float* d_images;
  CUDA_CHECK(cudaMalloc(&d_images, 28*28 * BATCH_SIZE * sizeof(float)));

  int* d_labels;
  CUDA_CHECK(cudaMalloc(&d_labels, BATCH_SIZE*sizeof(int)));


  cublasStatus_t state = cublasCreate(&handle);

  if (state != CUBLAS_STATUS_SUCCESS){
    printf("HANDLE CREATION");
  }

  int batches_per_epoch = count/EPOCHS;

  for (int iter=0; iter < EPOCHS; iter++){
    for (int batch=0; batch < batches_per_epoch; batch++){
        // Since the batches per epoch is just integer division, we remove the remaining images.
        int offset = batch * BATCH_SIZE;

        // Add starting address so that
        cudaMemcpy(&d_images, &images[(28*28*offset)], 28*28*BATCH_SIZE*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(&d_labels, &labels[offset], BATCH_SIZE*sizeof(int), cudaMemcpyHostToDevice);

        forward(&network, d_images, handle);

    }
  }

  float prob;

  cudaMemcpy(&prob, &network.post_act[NUM_LAYERS-1][0], sizeof(float), cudaMemcpyDeviceToHost);

  printf("Test probability %f", prob);


  free_nn(&network);
  cudaFree(d_images);
  cudaFree(d_labels);

  free_input(images, labels, count);
  return 0;
}

#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "cublas_v2.h"

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

#define BATCH_SIZE 10000

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
          if (i%in_dim == 0){
            h_b[i] = 1;
          }
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

void forward(NN* network, float* input){
    cublasHandle_t handle;
    float one = 1.0f;
    CUBLAS_CHECK(cublasCreate(&handle));
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
                        in, network->layers[l], &one, 
                        network->pre_act[l], network->layers[l+1]));
        dim3 blockDim(16,16);
        dim3 gridDim((network->layers[l+1] + blockDim.x - 1) / blockDim.x, 
                  (BATCH_SIZE + blockDim.y - 1) / blockDim.y);

        add_bias<<<gridDim, blockDim>>>(network->pre_act[l], network->bias[l], network->layers[l+1], BATCH_SIZE);

        relu_kernel<<<gridDim, blockDim>>>(network->pre_act[l], network->post_act[l],network->layers[l+1], BATCH_SIZE);
    }
}


void test_input(float* in) {
  float* d_in = (float*)malloc(784*2*sizeof(float));
  for (int i =0; i<784*2; i++){
    d_in[i] = 1;
  }
  CUDA_CHECK(cudaMalloc(&in, 784*2*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(in, d_in, 784 * 2 * sizeof(float),
                              cudaMemcpyHostToDevice));
}

int main() {
  NN network;
  network.layers[0] = 784;
  network.layers[1] = 100;
  network.layers[2] = 10;
  
  float* in;

  test_input(in);

  define_nn(&network);

  init_nn(&network);

  forward(&network, in);

  free_nn(&network);
  return 0;
}
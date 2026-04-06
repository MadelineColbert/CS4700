#include "nn.cuh"

void define_nn(NN* network){
    for (int l=0; l< NUM_LAYERS; l++){
        int in_dim  = network->layers[l];
        int out_dim = network->layers[l + 1];
        CUDA_CHECK(cudaMalloc(&network->weights[l], in_dim*out_dim*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&network->bias[l], out_dim*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&network->pre_act[l], BATCH_SIZE*out_dim*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&network->post_act[l], BATCH_SIZE*out_dim*sizeof(float)));
    }
    init_nn(network);
}

void init_nn(NN* network) {
    for (int l = 0; l < NUM_LAYERS; l++) {
        int in_dim  = network->layers[l];
        int out_dim = network->layers[l + 1];
        
        float* h_W = (float*)calloc(in_dim * out_dim, sizeof(float));
        float* h_b = (float*)calloc(out_dim, sizeof(float));
        CUDA_CHECK(cudaMemcpy(network->weights[l], h_W, in_dim * out_dim * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(network->bias[l], h_b, out_dim * sizeof(float),
                              cudaMemcpyHostToDevice));
        free(h_W); 
        free(h_b);
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

void load_data(int in, float** d_images, float** d_labels){
    cudaMalloc(d_images, BATCH_SIZE * in * sizeof(float));
    cudaMalloc(d_labels, BATCH_SIZE            * sizeof(float));
 
    // Read from disk into a host buffer, then copy to device
    float *h_images = (float*)malloc(BATCH_SIZE * in * sizeof(float));
    float *h_labels = (float*)malloc(BATCH_SIZE            * sizeof(float));
 
    FILE *f;
    f = fopen("mnist_data/test_images.bin", "rb");
    fread(h_images, sizeof(float), BATCH_SIZE * in, f); fclose(f);
    f = fopen("mnist_data/test_labels.bin", "rb");
    fread(h_labels, sizeof(float), BATCH_SIZE,            f); fclose(f);
 
    cudaMemcpy(*d_images, h_images, BATCH_SIZE * in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_labels, h_labels, BATCH_SIZE            * sizeof(float), cudaMemcpyHostToDevice);
    free(h_images);
    free(h_labels);
}

void free_input(float* d_images, float* d_labels) {
    CUDA_CHECK(cudaFree(d_images));
    CUDA_CHECK(cudaFree(d_labels));
}


__global__ void relu_forward_kernel(const float* __restrict__ Z,
                                    float*       __restrict__ A,
                                    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        A[i] = fmaxf(0.0f, Z[i]);
}
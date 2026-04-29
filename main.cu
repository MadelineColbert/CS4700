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

#define NUM_LAYERS 4

#define BATCH_SIZE 256
#define STEP_SIZE 0.05
#define EPOCHS 100
#define l2_regularizer 0.15

typedef struct{
    float* weights[NUM_LAYERS];
    float* bias[NUM_LAYERS];
    float *pre_act[NUM_LAYERS];
    float *post_act[NUM_LAYERS];
    float *pre_grad[NUM_LAYERS];
    float *post_grad[NUM_LAYERS];

    float *g[NUM_LAYERS];
    float *g_grad[NUM_LAYERS];
    float *post_g[NUM_LAYERS];

    float *w_grad[NUM_LAYERS];
    float *b_grad[NUM_LAYERS];
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

// This is really bad for now. Figure this out later.
__global__ void softmax_kernel(float* in, float* out, int length, int batch_size){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= batch_size) return;

    const float* src = in + col * length;
    float* dst = out + col * length;

    float mx = src[0];
    for (int i = 1; i < length; i++) {
        mx = fmaxf(mx, src[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < length; i++) {
        float val = expf(src[i] - mx);
        dst[i] = val;
        sum   += val;
    }

    for (int i = 0; i < length; i++) {
        dst[i] = dst[i] / sum;
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

__global__ void l2_regularization(float* params, float* g_grad, float* weights, float norm, float LR, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n){
        float final_g = g_grad[tid] * weights[tid];
        params[tid] -= LR*(final_g+norm);
        if (params[tid] < 0){
            params[tid] = 0;
        }
        if (params[tid] >1){
            params[tid] = 1;
        }
    }
}

__global__ void bias_grad_kernel(float* pre_act, float* bias, int out_dim, int batch){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;

    float acc = 0.0f;
    for (int col = 0; col < batch; col++)
        acc += pre_act[col * out_dim + row];
    bias[row] = acc;
}

__global__ void softmax_ce_kernel(float* prob_out, int* labels, float* pre_grad, int out_dim, int batch){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < out_dim && col < batch){
        int tid = col * out_dim + row;
        float correct_pred;
        if (labels[col] == row){
            correct_pred = 1.0f;
        } else {
            correct_pred = 0.0f;
        }
        pre_grad[tid] = prob_out[tid] - correct_pred;
    }
}


__global__ void relu_backward_kernel(const float* grad_out,
                                     const float* pre_act,
                                     float* grad_in,
                                     int out_dim, int batch)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < out_dim && col < batch) {
        int idx = col * out_dim + row;
        grad_in[idx] = grad_out[idx] * (pre_act[idx] > 0.0f ? 1.0f : 0.0f);
    }
}

__global__ void sgd_kernel(float* params, float* grad, float LR, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n){
        params[tid] -= LR * grad[tid];
    }
}

__global__ void gate_mult(float* weights, float* g, float* post_g, int in, int out ){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < out && col < in){
        int tid = out*col + row;
        if (g[tid] > 0.5){
            post_g[tid] = weights[tid];
        } else{
            post_g[tid] = 0;
        }
    }
}

__global__ void mask_w_grad(float* w_grad, float* g, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n){
        if (g[tid] < 0.5){
            w_grad[tid] = 0;
        }
    }
}

void init_nn(NN* network) {
    for (int l = 0; l < NUM_LAYERS; l++) {
        int in_dim  = network->layers[l];
        int out_dim = network->layers[l + 1];

        float xavier_init = sqrtf(6.0f/(in_dim+out_dim));

        float* h_W = (float*)calloc(in_dim * out_dim, sizeof(float));
        float* g = (float*)calloc(in_dim * out_dim, sizeof(float));
        float* h_b = (float*)calloc(out_dim, sizeof(float));
        for (int i=0; i<in_dim*out_dim; i++){
          h_W[i] = ((float)rand() / (float)RAND_MAX) * 2 * xavier_init - xavier_init;
          g[i] = ((float)rand() / (float)RAND_MAX);
        }
        CUDA_CHECK(cudaMemcpy(network->weights[l], h_W, in_dim * out_dim * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(network->bias[l], h_b, out_dim * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(network->g[l], g, in_dim * out_dim * sizeof(float),
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
        CUDA_CHECK(cudaMalloc(&network->w_grad[l], in_dim*out_dim*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&network->bias[l], out_dim*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&network->b_grad[l], out_dim*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&network->pre_act[l], BATCH_SIZE*out_dim*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&network->pre_grad[l], BATCH_SIZE*out_dim*sizeof(float)));

        CUDA_CHECK(cudaMalloc(&network->g[l], in_dim*out_dim*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&network->g_grad[l], in_dim*out_dim*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&network->post_g[l], in_dim*out_dim*sizeof(float)));

        CUDA_CHECK(cudaMalloc(&network->post_grad[l], BATCH_SIZE*out_dim*sizeof(float)));
        CUDA_CHECK(cudaMalloc(&network->post_act[l], BATCH_SIZE*out_dim*sizeof(float)));
    }
}

void free_nn(NN* network){
    for (int l=0; l< NUM_LAYERS; l++){
        CUDA_CHECK(cudaFree(network->weights[l]));
        CUDA_CHECK(cudaFree(network->bias[l]));
        CUDA_CHECK(cudaFree(network->pre_act[l]));
        CUDA_CHECK(cudaFree(network->post_act[l]));
        CUDA_CHECK(cudaFree(network->b_grad[l]));

        CUDA_CHECK(cudaFree(network->g[l]));
        CUDA_CHECK(cudaFree(network->g_grad[l]));
        CUDA_CHECK(cudaFree(network->post_g[l]));

        CUDA_CHECK(cudaFree(network->pre_grad[l]));
        CUDA_CHECK(cudaFree(network->post_grad[l]));
        CUDA_CHECK(cudaFree(network->w_grad[l]));
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

        dim3 gate_block(16,16);
        dim3 gate_grid((network->layers[l+1]+15)/16,
                (network->layers[l]+15)/16);
        gate_mult<<<gate_grid, gate_block>>>(network->weights[l], network->g[l], network->post_g[l], network->layers[l], network->layers[l+1]);

        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        network->layers[l+1], BATCH_SIZE, network->layers[l],
                        &one, network->post_g[l], network->layers[l+1],
                        in, network->layers[l], &zero,
                        network->pre_act[l], network->layers[l+1]));

        dim3 blockDim(16,16);
        dim3 gridDim((network->layers[l+1] + blockDim.x - 1) / blockDim.x,
                  (BATCH_SIZE + blockDim.y - 1) / blockDim.y);

        add_bias<<<gridDim, blockDim>>>(network->pre_act[l], network->bias[l], network->layers[l+1], BATCH_SIZE);

        if (l == NUM_LAYERS-1){
            // This is bad for now, figure out later...
            int threads = 256;
            int blocks = (BATCH_SIZE + threads - 1) / threads;
            softmax_kernel<<<blocks, threads>>>(network->pre_act[l], network->post_act[l], network->layers[l+1], BATCH_SIZE);
        } else {
            relu_kernel<<<gridDim, blockDim>>>(network->pre_act[l], network->post_act[l],network->layers[l+1], BATCH_SIZE);
        }
    }
}

void backward(NN* network, float* input, int* labels, cublasHandle_t handle){
    float one = 1.0f;
    float zero= 0.0f;
    float* in;

    dim3 block_256(256);
    dim3 block_ce(16, 16);
    dim3 grid_ce((network->layers[NUM_LAYERS] + block_ce.x - 1) / block_ce.x,
                (BATCH_SIZE + block_ce.y - 1) / block_ce.y);

    // Start by initializing with cross entropy loss (Divided by batch size is handled here)
    softmax_ce_kernel<<<grid_ce, block_ce>>>(network->post_act[NUM_LAYERS-1], labels, network->pre_grad[NUM_LAYERS-1], network->layers[NUM_LAYERS], BATCH_SIZE);

    for (int l=NUM_LAYERS-1; l>=0; l--){
        int w_size = network->layers[l] * network->layers[l+1];
        dim3 w_grid((w_size+255)/256);
        dim3 b_grid((255+network->layers[l+1])/256);
        if (l == 0) {
            in = input;
        }
        else {
            in = network->post_act[l-1];
        }

        // Gradient of W = gradient of pre_activation * input^T
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                network->layers[l+1], network->layers[l], BATCH_SIZE,
                                &one, network->pre_grad[l], network->layers[l+1],
                                in, network->layers[l], &zero,
                                network->w_grad[l], network->layers[l+1]));

        l2_regularization<<<w_grid, block_256>>>(network->g[l], network->w_grad[l], network->weights[l], l2_regularizer, STEP_SIZE/(float)BATCH_SIZE, w_size);
        mask_w_grad<<<w_grid, block_256>>>(network->w_grad[l], network->g[l], w_size);

        // Sum over the batch of pre-gradients to get the bias gradient
        dim3 b_grad_block(256);
        dim3 b_grad_grid((network->layers[l+1] + 255) / 256);
        bias_grad_kernel<<<b_grad_grid, b_grad_block>>>(
                network->pre_grad[l], network->b_grad[l], network->layers[l+1], BATCH_SIZE);

        if (l > 0){
            int prev_out = network->layers[l];

            // Backpropagate past the weights into the previous layer
            CUBLAS_CHECK(cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                network->layers[l], BATCH_SIZE, network->layers[l+1],
                &one,  network->post_g[l], network->layers[l+1],
                       network->pre_grad[l], network->layers[l+1],
                &zero, network->post_grad[l-1],   network->layers[l]));


            // Handle the ReLU activation gradient from pre-act to post-act
            dim3 block(16, 16);
            dim3 grid((prev_out   + 15) / 16,
                      (BATCH_SIZE + 15) / 16);
            relu_backward_kernel<<<grid, block>>>(
                network->post_grad[l-1], network->pre_act[l - 1], network->pre_grad[l - 1],
                prev_out, BATCH_SIZE);

        }

        // SGD Parameter Update
        sgd_kernel<<<w_grid, block_256>>>(network->weights[l], network->w_grad[l], STEP_SIZE/(float)BATCH_SIZE, w_size);
        
        sgd_kernel<<<b_grid, block_256>>>(network->bias[l], network->b_grad[l], STEP_SIZE/(float)BATCH_SIZE, network->layers[l+1]);
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

float compute_loss(const float* d_probs, const int* d_labels,
                   int out_dim)
{
    size_t probs = (size_t)out_dim * BATCH_SIZE * sizeof(float);
    size_t labels = (size_t)BATCH_SIZE           * sizeof(int);

    float* h_probs  = (float*)malloc(probs);
    int*   h_labels = (int*)  malloc(labels);

    CUDA_CHECK(cudaMemcpy(h_probs,  d_probs,  probs, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_labels, d_labels, labels, cudaMemcpyDeviceToHost));

    float loss = 0.0f;
    for (int j = 0; j < BATCH_SIZE; j++) {
        int   c = h_labels[j];
        float p = h_probs[j * out_dim + c];
        // Cross-Entropy Loss
        loss   -= logf(fmaxf(p, 1e-7f));
    }

    free(h_probs);
    free(h_labels);
    return loss / (float)BATCH_SIZE;
}

int main() {
  NN network;
  cublasHandle_t handle;
  network.layers[0] = 784;
  network.layers[1] = 100;
  network.layers[2] = 100;
  network.layers[3] = 100;
  network.layers[4] = 10;

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

  int batches_per_epoch = count/BATCH_SIZE;

  for (int iter=0; iter < EPOCHS; iter++){
    for (int batch=0; batch < batches_per_epoch; batch++){
        // Since the batches per epoch is just integer division, we remove the remaining images.
        int offset = batch * BATCH_SIZE;

        // Add starting address so that
        cudaMemcpy(d_images, &images[(28*28*offset)], 28*28*BATCH_SIZE*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_labels, &labels[offset], BATCH_SIZE*sizeof(int), cudaMemcpyHostToDevice);

        forward(&network, d_images, handle);
        backward(&network, d_images, d_labels, handle);
        float loss = compute_loss(
                    network.post_act[NUM_LAYERS - 1],
                    d_labels,
                    network.layers[NUM_LAYERS]);
        printf("Epoch %d  batch %d/%d  loss = %f\n",
                       iter + 1, batch, batches_per_epoch, loss);
    }
  }

  float* gates;

  gates = (float*)malloc(28*28*100*sizeof(float));

  CUDA_CHECK(cudaMemcpy(gates, network.g[0], 28*28*100*sizeof(float), cudaMemcpyDeviceToHost));

  for (int i =0; i < 28*28*100; i++){
    printf("%f \n", gates[i]);
  }

  free_nn(&network);
  cudaFree(d_images);
  cudaFree(d_labels);

  free_input(images, labels, count);
  return 0;
}
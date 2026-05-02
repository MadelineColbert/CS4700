#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include <cusparse.h>


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

#define CUSPARSE_CHECK(call)                                              \
    do {                                                                  \
        cusparseStatus_t _s = (call);                                     \
        if (_s != CUSPARSE_STATUS_SUCCESS) {                              \
            fprintf(stderr, "cuSPARSE error %s:%d — code %d\n",           \
                    __FILE__, __LINE__, (int)_s);                         \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

#define NUM_LAYERS 4

#define BATCH_SIZE 256
#define STEP_SIZE 0.05
#define EPOCHS 10
#define l2_regularizer 0.15
#define IMAGE_LENGTH 28
#define OUTPUT_LAYER_SIZE 10
#define INTERNAL_LAYER_SIZE 100

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

typedef enum { LAYER_SPARSE, LAYER_DENSE } LayerType;

typedef struct {
    LayerType type[NUM_LAYERS];

    float *csrVal[NUM_LAYERS];
    int *csrRowPtr[NUM_LAYERS];
    int *csrColInd[NUM_LAYERS];
    cusparseSpMatDescr_t matDescr[NUM_LAYERS];

    float *denseW[NUM_LAYERS];

    float *bias[NUM_LAYERS];
    int in_dims[NUM_LAYERS];
    int out_dims[NUM_LAYERS];
    int nnz[NUM_LAYERS];
} PostProcessedNN;

enum Pruning {
    PRUNING,
    NO_PRUNE
};

enum Gates{
    GATES,
    NO_GATES
};

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

__global__ void regularization(float* params, float* g_grad, float* weights, float norm, float LR, int n){
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

__global__ void l2_pruning(float* weights, float threshold, int neurons, int inputs){
    extern __shared__ float neuron_sums[];

    int neuron = blockIdx.x;
    int tid = threadIdx.x;

    if (neuron > neurons){
        return;
    }

    int starting_idx = neuron * inputs;
    float thread_sum =0;

    for (int i=tid; i < inputs; i++){
        float curr_w = weights[i+starting_idx];
        thread_sum += curr_w * curr_w;
    }

    neuron_sums[tid] = thread_sum;
    __syncthreads();

    for (int tds = blockDim.x/2; tds > 0; tds/=2){
        if (tid < tds){
            neuron_sums[tid] += neuron_sums[tid+tds]; 
        }
        __syncthreads();
    }

    __shared__ bool prune;
    if (tid == 0){
        float l2_norm = sqrtf(neuron_sums[0]);
        prune = (l2_norm < threshold);
    }

    if (prune) {
        for (int i=tid; i < inputs; i++){
            weights[i+starting_idx] = 0;
        }
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

__global__ void sgd_kernel(float* params, float* grad, float LR, int n, float decay){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n){
        params[tid] = (1-LR*decay)*params[tid] - (LR * grad[tid]);
    }
}

__global__ void remove_gates(float* weights, float* g, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n){
        if (g[tid] > 0.5){
            weights[tid] = 0;
        }
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

void free_optimized_nn(PostProcessedNN* optNet) {
    for (int l = 0; l < NUM_LAYERS; l++) {
        // Shared Bias
        if (optNet->bias[l]) CUDA_CHECK(cudaFree(optNet->bias[l]));

        if (optNet->type[l] == LAYER_SPARSE) {
            // Sparse-specific cleanup
            if (optNet->csrVal[l])    CUDA_CHECK(cudaFree(optNet->csrVal[l]));
            if (optNet->csrRowPtr[l]) CUDA_CHECK(cudaFree(optNet->csrRowPtr[l]));
            if (optNet->csrColInd[l]) CUDA_CHECK(cudaFree(optNet->csrColInd[l]));
            if (optNet->matDescr[l])  CUSPARSE_CHECK(cusparseDestroySpMat(optNet->matDescr[l]));
        } else {
            // Dense-specific cleanup
            if (optNet->denseW[l])    CUDA_CHECK(cudaFree(optNet->denseW[l]));
        }
    }
}

void forward(NN* network, float* input, cublasHandle_t handle, Gates gate){
    float one = 1.0f;
    float zero= 0.0f;
    float* in;
    for (int l=0; l<NUM_LAYERS; l++) {
        if (l == 0){
            in = input;
        } else{
            in = network->post_act[l-1];
        }

        if (gate == GATES) {
            dim3 gate_block(16,16);
            dim3 gate_grid((network->layers[l+1]+15)/16,
                    (network->layers[l]+15)/16);
            gate_mult<<<gate_grid, gate_block>>>(network->weights[l], network->g[l], network->post_g[l], network->layers[l], network->layers[l+1]);

            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            network->layers[l+1], BATCH_SIZE, network->layers[l],
                            &one, network->post_g[l], network->layers[l+1],
                            in, network->layers[l], &zero,
                            network->pre_act[l], network->layers[l+1]));
        }
        else {
            CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        network->layers[l+1], BATCH_SIZE, network->layers[l],
                        &one, network->weights[l], network->layers[l+1],
                        in, network->layers[l], &zero,
                        network->pre_act[l], network->layers[l+1]));
        }

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

void backward(NN* network, float* input, int* labels, cublasHandle_t handle, float decay, Gates gate){
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

        if (gate == GATES){
            regularization<<<w_grid, block_256>>>(network->g[l], network->w_grad[l], network->weights[l], l2_regularizer, STEP_SIZE/(float)BATCH_SIZE, w_size);
            mask_w_grad<<<w_grid, block_256>>>(network->w_grad[l], network->g[l], w_size);
        }

        // Sum over the batch of pre-gradients to get the bias gradient
        dim3 b_grad_block(256);
        dim3 b_grad_grid((network->layers[l+1] + 255) / 256);
        bias_grad_kernel<<<b_grad_grid, b_grad_block>>>(
                network->pre_grad[l], network->b_grad[l], network->layers[l+1], BATCH_SIZE);

        if (l > 0){
            int prev_out = network->layers[l];

            // Backpropagate past the weights into the previous layer
            if (gate == GATES){
                 CUBLAS_CHECK(cublasSgemm(handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        network->layers[l], BATCH_SIZE, network->layers[l+1],
                        &one,  network->post_g[l], network->layers[l+1],
                            network->pre_grad[l], network->layers[l+1],
                        &zero, network->post_grad[l-1],   network->layers[l]));
            } else {
                CUBLAS_CHECK(cublasSgemm(handle,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        network->layers[l], BATCH_SIZE, network->layers[l+1],
                        &one,  network->weights[l], network->layers[l+1],
                            network->pre_grad[l], network->layers[l+1],
                        &zero, network->post_grad[l-1],   network->layers[l]));
            }
           


            // Handle the ReLU activation gradient from pre-act to post-act
            dim3 block(16, 16);
            dim3 grid((prev_out   + 15) / 16,
                      (BATCH_SIZE + 15) / 16);
            relu_backward_kernel<<<grid, block>>>(
                network->post_grad[l-1], network->pre_act[l - 1], network->pre_grad[l - 1],
                prev_out, BATCH_SIZE);

        }

        // SGD Parameter Update
        sgd_kernel<<<w_grid, block_256>>>(network->weights[l], network->w_grad[l], STEP_SIZE/(float)BATCH_SIZE, w_size, decay);
        
        sgd_kernel<<<b_grid, block_256>>>(network->bias[l], network->b_grad[l], STEP_SIZE/(float)BATCH_SIZE, network->layers[l+1], decay);
    }
}

void post_processing(NN* network, float threshold){
    for (int l =0; l < NUM_LAYERS; l++){

        int threads = 256;
        int blocks = ((network->layers[l] * network->layers[l+1]) + threads - 1) / threads;
        remove_gates<<<blocks, threads>>>(network->weights[l], network->g[l], network->layers[l] * network->layers[l+1]);

        l2_pruning<<<network->layers[l+1], 256, 256 * sizeof(float)>>>(
            network->weights[l], threshold, network->layers[l+1], network->layers[l]);
    }
}

void convert_to_optimized(NN* oldNet, PostProcessedNN* optNet, float dense_cutoff) {
    cusparseHandle_t spHandle;
    CUSPARSE_CHECK(cusparseCreate(&spHandle));

    for (int l = 0; l < NUM_LAYERS; l++) {
        int orig_in = oldNet->layers[l];
        int orig_out = oldNet->layers[l+1];
        
        float* h_W = (float*)malloc(orig_in * orig_out * sizeof(float));
        float* h_b = (float*)malloc(orig_out * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_W, oldNet->weights[l], orig_in * orig_out * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_b, oldNet->bias[l], orig_out * sizeof(float), cudaMemcpyDeviceToHost));

        // 1. Identify "Live" Neurons (Remove rows that are all zero)
        int live_neurons = 0;
        int* live_indices = (int*)malloc(orig_out * sizeof(int));
        for (int r = 0; r < orig_out; r++) {
            bool is_dead = true;
            for (int c = 0; c < orig_in; c++) {
                if (fabsf(h_W[c * orig_out + r]) > 1e-9f) {
                    is_dead = false;
                    break;
                }
            }
            if (!is_dead) live_indices[live_neurons++] = r;
        }

        optNet->in_dims[l] = orig_in;
        optNet->out_dims[l] = live_neurons;

        // 2. Extract live weights into a temporary compact buffer
        float* h_compactW = (float*)malloc(orig_in * live_neurons * sizeof(float));
        float* h_compactB = (float*)malloc(live_neurons * sizeof(float));
        int total_nnz = 0;

        for (int r_idx = 0; r_idx < live_neurons; r_idx++) {
            int original_row = live_indices[r_idx];
            h_compactB[r_idx] = h_b[original_row];
            for (int c = 0; c < orig_in; c++) {
                float val = h_W[c * orig_out + original_row];
                h_compactW[c * live_neurons + r_idx] = val;
                if (fabsf(val) > 1e-9f) total_nnz++;
            }
        }

        float density = (float)total_nnz / (orig_in * live_neurons);
        optNet->nnz[l] = total_nnz;
        CUDA_CHECK(cudaMalloc(&optNet->bias[l], live_neurons * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(optNet->bias[l], h_compactB, live_neurons * sizeof(float), cudaMemcpyHostToDevice));

        // 3. Choose Sparse vs Dense
        if (density > dense_cutoff) {
            optNet->type[l] = LAYER_DENSE;
            CUDA_CHECK(cudaMalloc(&optNet->denseW[l], orig_in * live_neurons * sizeof(float)));
            CUDA_CHECK(cudaMemcpy(optNet->denseW[l], h_compactW, orig_in * live_neurons * sizeof(float), cudaMemcpyHostToDevice));
        } else {
            optNet->type[l] = LAYER_SPARSE;
            // Build CSR from compact weights
            float* h_csrVal = (float*)malloc(total_nnz * sizeof(float));
            int* h_csrColInd = (int*)malloc(total_nnz * sizeof(int));
            int* h_csrRowPtr = (int*)malloc((live_neurons + 1) * sizeof(int));
            int curr_nnz = 0;
            for (int r = 0; r < live_neurons; r++) {
                h_csrRowPtr[r] = curr_nnz;
                for (int c = 0; c < orig_in; c++) {
                    float val = h_compactW[c * live_neurons + r];
                    if (fabsf(val) > 1e-9f) {
                        h_csrVal[curr_nnz] = val;
                        h_csrColInd[curr_nnz] = c;
                        curr_nnz++;
                    }
                }
            }
            h_csrRowPtr[live_neurons] = curr_nnz;

            CUDA_CHECK(cudaMalloc(&optNet->csrVal[l], total_nnz * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&optNet->csrColInd[l], total_nnz * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&optNet->csrRowPtr[l], (live_neurons + 1) * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(optNet->csrVal[l], h_csrVal, total_nnz * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(optNet->csrColInd[l], h_csrColInd, total_nnz * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(optNet->csrRowPtr[l], h_csrRowPtr, (live_neurons + 1) * sizeof(int), cudaMemcpyHostToDevice));

            CUSPARSE_CHECK(cusparseCreateCsr(&optNet->matDescr[l], live_neurons, orig_in, total_nnz,
                                             optNet->csrRowPtr[l], optNet->csrColInd[l], optNet->csrVal[l],
                                             CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
            free(h_csrVal); free(h_csrColInd); free(h_csrRowPtr);
        }

        free(h_W); free(h_b); free(h_compactW); free(h_compactB); free(live_indices);
    }
    cusparseDestroy(spHandle);
}

void optimized_forward(PostProcessedNN* optNet, float* d_input, float** act_buffers, cublasHandle_t cbHandle, cusparseHandle_t spHandle) {
    float one = 1.0f, zero = 0.0f;

    for (int l = 0; l < NUM_LAYERS; l++) {
        float* in = (l == 0) ? d_input : act_buffers[l-1];
        float* out = act_buffers[l];

        if (optNet->type[l] == LAYER_DENSE) {
            CUBLAS_CHECK(cublasSgemm(cbHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     optNet->out_dims[l], BATCH_SIZE, optNet->in_dims[l],
                                     &one, optNet->denseW[l], optNet->out_dims[l],
                                     in, optNet->in_dims[l], &zero, out, optNet->out_dims[l]));
        } else {
            cusparseDnMatDescr_t matIn, matOut;
            cusparseCreateDnMat(&matIn, optNet->in_dims[l], BATCH_SIZE, optNet->in_dims[l], in, CUDA_R_32F, CUSPARSE_ORDER_COL);
            cusparseCreateDnMat(&matOut, optNet->out_dims[l], BATCH_SIZE, optNet->out_dims[l], out, CUDA_R_32F, CUSPARSE_ORDER_COL);
            
            size_t bufSize = 0;
            cusparseSpMM_bufferSize(spHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &one, optNet->matDescr[l], matIn, &zero, matOut, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufSize);
            void* d_buf; cudaMalloc(&d_buf, bufSize);
            cusparseSpMM(spHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         &one, optNet->matDescr[l], matIn, &zero, matOut, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_buf);
            
            cudaFree(d_buf);
            cusparseDestroyDnMat(matIn); cusparseDestroyDnMat(matOut);
        }

        dim3 block(16, 16), grid((optNet->out_dims[l] + 15)/16, (BATCH_SIZE + 15)/16);
        add_bias<<<grid, block>>>(out, optNet->bias[l], optNet->out_dims[l], BATCH_SIZE);
        if (l == NUM_LAYERS-1) softmax_kernel<<<(BATCH_SIZE+255)/256, 256>>>(out, out, optNet->out_dims[l], BATCH_SIZE);
        else relu_kernel<<<grid, block>>>(out, out, optNet->out_dims[l], BATCH_SIZE);
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
    *images = (float*)malloc(sizeof(float)*image_cnt*IMAGE_LENGTH*IMAGE_LENGTH);
    *labels = (int*)malloc(sizeof(int)*image_cnt);

    for (int i=0; i< image_cnt; i++) {
        unsigned char read_data[IMAGE_LENGTH*IMAGE_LENGTH];
        fread(read_data, 1, IMAGE_LENGTH*IMAGE_LENGTH, ifp);
        for (int j=0; j < IMAGE_LENGTH*IMAGE_LENGTH; j++){
            (*images)[i*IMAGE_LENGTH*IMAGE_LENGTH + j] = read_data[j] / 255.0f;
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

void test_nn(NN* network, cublasHandle_t handle){
    float* images = NULL;
    int* labels=NULL;
    int image_count=0;

    load_mnist("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", &images, &labels, &image_count);

    float* d_images;
    CUDA_CHECK(cudaMalloc(&d_images, IMAGE_LENGTH*IMAGE_LENGTH * BATCH_SIZE * sizeof(float)));

    int batches_per_iter = image_count/BATCH_SIZE;
    int correct_pred_count = 0;

    for (int batch=0; batch < batches_per_iter; batch++){
        int offset = batch * BATCH_SIZE;
        
        cudaMemcpy(d_images, &images[(IMAGE_LENGTH*IMAGE_LENGTH*offset)], IMAGE_LENGTH*IMAGE_LENGTH*BATCH_SIZE*sizeof(float), cudaMemcpyHostToDevice);

        forward(network, d_images, handle);

        float* h_outputs = (float*)malloc(BATCH_SIZE * OUTPUT_LAYER_SIZE * sizeof(float));

        cudaMemcpy(h_outputs, network->post_act[NUM_LAYERS-1], BATCH_SIZE * OUTPUT_LAYER_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
        
        for(int image=0; image < BATCH_SIZE; image++){
            int image_offset = image * OUTPUT_LAYER_SIZE;

            float maxOutput = h_outputs[image_offset];
            int maxOutputIndex = 0;

            for(int outputIndex=1; outputIndex < OUTPUT_LAYER_SIZE; outputIndex++){
                float output = h_outputs[image_offset + outputIndex];;

                if(output > maxOutput){
                    maxOutput = output;
                    maxOutputIndex = outputIndex;
                }
            }

            if(labels[offset+image] == maxOutputIndex){
                correct_pred_count++;
            }
        }
    }

    float percentage = correct_pred_count / ((float) image_count) * 100.0f; 
    printf("%d/%d (%.2f%%) correctly predicted\n", correct_pred_count, image_count, percentage);

    cudaFree(d_images);
    free_input(images, labels, image_count);
}

void train(float decay, Gates gate, float threshold){
  for(int k=0; k<10; k++){
    printf("Iteration %d:\n",k);
        NN network;
        cublasHandle_t handle;
        network.layers[0] = IMAGE_LENGTH * IMAGE_LENGTH;
        network.layers[1] = INTERNAL_LAYER_SIZE;
        network.layers[2] = INTERNAL_LAYER_SIZE;
        network.layers[3] = INTERNAL_LAYER_SIZE;
        network.layers[4] = OUTPUT_LAYER_SIZE;
        float* images = NULL;
        int* labels=NULL;
        int count=0;

        load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte", &images, &labels, &count);

        define_nn(&network);

        init_nn(&network);

    float* d_images;
    CUDA_CHECK(cudaMalloc(&d_images, IMAGE_LENGTH*IMAGE_LENGTH * BATCH_SIZE * sizeof(float)));

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
            cudaMemcpy(d_images, &images[(IMAGE_LENGTH*IMAGE_LENGTH*offset)], IMAGE_LENGTH*IMAGE_LENGTH*BATCH_SIZE*sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_labels, &labels[offset], BATCH_SIZE*sizeof(int), cudaMemcpyHostToDevice);

            forward(&network, d_images, handle, gate);
            backward(&network, d_images, d_labels, handle, decay, gate);
            float loss = compute_loss(
                        network.post_act[NUM_LAYERS - 1],
                        d_labels,
                        network.layers[NUM_LAYERS]);
            //printf("Epoch %d  batch %d/%d  loss = %f\n", iter + 1, batch, batches_per_epoch, loss);
        }
    }

    test_nn(&network, handle);

    post_processing(&network, threshold);

    // 2. Initialize and Convert to Optimized Network
    PostProcessedNN optNet;
    // Zero out pointers initially to prevent freeing garbage if conversion fails
    memset(&optNet, 0, sizeof(PostProcessedNN)); 
    
    convert_to_optimized(&network, &optNet, 0.3);

    // 3. Prepare for Optimized Inference
    cusparseHandle_t spHandle;
    cusparseCreate(&spHandle);
    
    // We need new activation buffers because neuron counts (out_dims) may have changed
    float* opt_act_buffers[NUM_LAYERS];
    for (int l = 0; l < NUM_LAYERS; l++) {
        CUDA_CHECK(cudaMalloc(&opt_act_buffers[l], BATCH_SIZE * optNet.out_dims[l] * sizeof(float)));
    }

    // 4. Run Inference on Test Set (Example)
    // Assume d_test_images is loaded
    optimized_forward(&optNet, d_images, opt_act_buffers, handle, spHandle);

    // 5. Cleanup
    for (int l = 0; l < NUM_LAYERS; l++) {
        CUDA_CHECK(cudaFree(opt_act_buffers[l]));
    }
    cusparseDestroy(spHandle);
    free_optimized_nn(&optNet);


    free_nn(&network);
    cudaFree(d_images);
    cudaFree(d_labels);
    cublasDestroy(handle);

    free_input(images, labels, count);
    }
}

int main() {
    printf("====GATES NO DECAY====");
    train(0.0f, GATES, 1.0f);
    printf("====GATES DECAY====");
    train(0.001f, GATES, 1.0f);
    printf("====NO GATES NO DECAY====");
    train(0.0f, NO_GATES, 1.0f);
    printf("====NO GATES DECAY====");
    train(0.001f, NO_GATES, 1.0f);
    return 0;
}
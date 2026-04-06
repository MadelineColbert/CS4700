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
    for (int l = 0; l < network->layers; l++) {
        int in_dim  = network->layers[l];
        int out_dim = network->layers[l + 1];
        
        float* h_W = (float*)calloc(in_dim * out_dim * sizeof(float));
        float* h_b = (float*)calloc(out_dim, sizeof(float));
        CUDA_CHECK(cudaMemcpy(network->weights[l], h_W, in_dim * out_dim * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(network->bias[l], h_b, out_dim * sizeof(float),
                              cudaMemcpyHostToDevice));
        free(h_W); 
        free(h_b);
    }
}

void free_nn(NN* nn){
    for (int l=0; l< NUM_LAYERS; l++){
        CUDA_CHECK(cudaFree(network->weights[l]));
        CUDA_CHECK(cudaFree(network->bias[l]));
        CUDA_CHECK(cudaFree(network->pre_act[l]));
        CUDA_CHECK(cudaFree(network->post_act[l]));
    }
}

void load_data(int in, float* d_images, float* d_labels){
    cudaMalloc(&d_images, BATCH_SIZE * in * sizeof(float));
    cudaMalloc(&d_labels, BATCH_SIZE            * sizeof(float));
 
    // Read from disk into a host buffer, then copy to device
    float *h_images = (float*)malloc(BATCH_SIZE * in * sizeof(float));
    float *h_labels = (float*)malloc(BATCH_SIZE            * sizeof(float));
 
    FILE *f;
    f = fopen("mnist_data/test_images.bin", "rb");
    fread(h_images, sizeof(float), BATCH_SIZE * in, f); fclose(f);
    f = fopen("mnist_data/test_labels.bin", "rb");
    fread(h_labels, sizeof(float), BATCH_SIZE,            f); fclose(f);
 
    cudaMemcpy(d_images, h_images, BATCH_SIZE * in * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, BATCH_SIZE            * sizeof(float), cudaMemcpyHostToDevice);
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

float* forward_pass(NN* nn, float* input, cublasHandle_t handle) {
    const float alpha = 1.0f, beta = 0.0f;

    for (int l = 0; l < NUM_LAYERS; l++) {
        int in_dim  = nn->layers[l];
        int out_dim = nn->layers[l + 1];
        int total   = out_dim * BATCH_SIZE;

        // A_prev is X_input for layer 0, else A[l-1]
        float* A_prev = (l == 0) ? net->X_input : net->A[l - 1];

        /*
         * Z[l] = W[l] · A_prev
         *
         * cuBLAS uses column-major, so we compute:
         *   Z (out_dim × batch) = W (out_dim × in_dim) · A_prev (in_dim × batch)
         *
         * cublasSgemm args: (handle, transA, transB, M, N, K, α, A, lda, B, ldb, β, C, ldc)
         *   M = out_dim, N = BATCH_SIZE, K = in_dim
         */
        CUBLAS_CHECK(cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            out_dim, BATCH_SIZE, in_dim,
            &alpha,
            net->W[l],  out_dim,   // W: [out_dim, in_dim]
            A_prev,     in_dim,    // A_prev: [in_dim, batch]
            &beta,
            net->Z[l],  out_dim)); // Z: [out_dim, batch]

        // Add bias to every column
        int blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
        bias_forward_kernel<<<blocks, BLOCK_SIZE>>>(
            net->pre_act[l], net->b[l], out_dim, BATCH_SIZE);

        // ReLU activation (skip for output layer — raw logits for loss)
        if (l < NUM_LAYERS - 1) {
            relu_forward_kernel<<<blocks, BLOCK_SIZE>>>(
                net->pre_act[l], net->post_act[l], total);
        } else {
            // Output layer: A = Z (identity / linear)
            CUDA_CHECK(cudaMemcpy(net->post_act[l], net->pre_act[l],
                                  total * sizeof(float),
                                  cudaMemcpyDeviceToDevice));
        }
    }
}
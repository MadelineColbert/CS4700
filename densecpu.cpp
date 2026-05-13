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
#define ITERATIONS 3

int mnist_bin_to_int(char* v) {
    int i;
    int ret = 0;
    for (i = 0; i < 4; i++) {
        ret <<= 8;
        ret |= (unsigned char)v[i];
    }
    return ret;
}

void load_mnist(const char* image_file, const char* label_file,
                float** images, int** labels, int* count) {
    FILE* ifp = fopen(image_file, "rb");
    FILE* lfp = fopen(label_file, "rb");

    if (!ifp || !lfp) {
        printf("Failed to open MNIST files:\n%s\n%s\n",
            image_file, label_file);
        exit(1);
    }
    char tmp[4];

    fread(tmp, 1, 4, ifp);
    fread(tmp, 1, 4, lfp);

    fread(tmp, 1, 4, ifp);
    int image_cnt = mnist_bin_to_int(tmp);

    fread(tmp, 1, 4, lfp);

    fread(tmp, 1, 4, ifp);
    fread(tmp, 1, 4, ifp);

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

void free_input(float* images, int* labels) {
    free(images);
    free(labels);
}

void log_result(float acc,
                double train_time,
                double inference_time)
{
    FILE* f = fopen("results.csv", "a");

    fseek(f, 0, SEEK_END);

    if (ftell(f) == 0) {
        fprintf(f,
            "step_size,internal_size,mode,accuracy,time,nnzs,test_time,threshold,decay,regularizer\n");
    }

    fprintf(f,
        "0.000000,-1,CPU,%.4f,%.4f,0,%.4f,0.0000,0.0000,0.0000\n",
        acc,
        train_time,
        inference_time);

    fclose(f);
}

float test_nn(Network* neuralNet) {
    float* images = NULL;
    int* labels = NULL;
    int count = 0;

    load_mnist("t10k-images-idx3-ubyte",
               "t10k-labels-idx1-ubyte",
               &images,
               &labels,
               &count);

    int correct = 0;

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < count; i++) {
        RowVector inputVec =
            Eigen::Map<RowVector>(images + i * 784, 784);

        neuralNet->forward(&inputVec);

        RowVector* results = neuralNet->getResults();

        int best_idx = 0;
        float best_val = (*results)(0);

        for (int j = 1; j < 10; j++) {
            if ((*results)(j) > best_val) {
                best_val = (*results)(j);
                best_idx = j;
            }
        }

        if (best_idx == labels[i]) {
            correct++;
        }
    }

    auto end = std::chrono::steady_clock::now();

    double inference_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            end - start).count();

    float accuracy =
        (float)correct / (float)count * 100.0f;

    printf("%d/%d (%.2f%%) correctly predicted\n",
           correct,
           count,
           accuracy);

    free_input(images, labels);

    return inference_time;
}

int main() {
    double train_times[ITERATIONS];
    double inference_times[ITERATIONS];
    float accuracies[ITERATIONS];

    for (int run = 0; run < ITERATIONS; run++) {
        int layerConfig[] = {
            IMAGE_NEURONS,
            LAYER_1,
            LAYER_2,
            LAYER_3,
            RESULT_LAYER
        };

        Network* neuralNet = new Network(5, layerConfig);

        float* images = NULL;
        int* labels = NULL;
        int count = 0;

        load_mnist("train-images-idx3-ubyte",
                   "train-labels-idx1-ubyte",
                   &images,
                   &labels,
                   &count);

        int batches_per_epoch = count / BATCH_SIZE;

        auto train_start = std::chrono::steady_clock::now();

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            for (int batch = 0; batch < batches_per_epoch; batch++) {

                int offset = batch * BATCH_SIZE;

                for (int i = 0; i < BATCH_SIZE; i++) {

                    RowVector inputVec =
                        Eigen::Map<RowVector>(
                            images + (offset + i) * 784,
                            784);

                    int label = labels[offset + i];

                    RowVector targetVec(10);
                    targetVec.setZero();
                    targetVec(label) = 1.0f;

                    neuralNet->forward(&inputVec);
                    neuralNet->backward(&targetVec);
                }
            }
        }

        auto train_end = std::chrono::steady_clock::now();

        train_times[run] =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                train_end - train_start).count();

        printf("Training Time: %.3f ms\n", train_times[run]);

        auto inference_start = std::chrono::steady_clock::now();

        float* test_images = NULL;
        int* test_labels = NULL;
        int test_count = 0;

        load_mnist("t10k-images-idx3-ubyte",
                   "t10k-labels-idx1-ubyte",
                   &test_images,
                   &test_labels,
                   &test_count);

        int correct = 0;

        for (int i = 0; i < test_count; i++) {

            RowVector inputVec =
                Eigen::Map<RowVector>(
                    test_images + i * 784,
                    784);

            neuralNet->forward(&inputVec);

            RowVector* results = neuralNet->getResults();

            int best_idx = 0;
            float best_val = (*results)(0);

            for (int j = 1; j < 10; j++) {
                if ((*results)(j) > best_val) {
                    best_val = (*results)(j);
                    best_idx = j;
                }
            }

            if (best_idx == test_labels[i]) {
                correct++;
            }
        }

        auto inference_end = std::chrono::steady_clock::now();

        inference_times[run] =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                inference_end - inference_start).count();

        accuracies[run] =
            (float)correct / (float)test_count * 100.0f;

        printf("Accuracy: %.2f%%\n", accuracies[run]);
        printf("Inference Time: %.3f ms\n", inference_times[run]);

        free_input(images, labels);
        free_input(test_images, test_labels);

        delete neuralNet;
    }

    double avg_train = 0.0;
    double avg_infer = 0.0;
    float avg_acc = 0.0f;

    for (int i = 0; i < ITERATIONS; i++) {
        avg_train += train_times[i];
        avg_infer += inference_times[i];
        avg_acc += accuracies[i];
    }

    avg_train /= ITERATIONS;
    avg_infer /= ITERATIONS;
    avg_acc /= ITERATIONS;

    log_result(avg_acc, avg_train, avg_infer);

    return 0;
}
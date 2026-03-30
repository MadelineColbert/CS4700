#include <stdio.h>
#include "src/nn.cuh"

int main() {
  int layers[1] = {3};
  NN* nn = define_nn(2,2, layers, 1);
  printf("Number of inputs and outputs: %d %d\n", nn->inputs, nn->outputs);
  float input[2] = {2,3};
  float* output = forward_pass(nn, input);
  for (int i =0; i<2; i++) {
    printf("Current Output: %f\n", output[i]);
  }
  free_nn(nn);
  return 0;
}
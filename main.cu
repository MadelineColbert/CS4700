#include <stdio.h>
#include "src/test.cuh"
#include "src/nn.cuh"

int main() {
  int layers[1] = {2};
  NN* nn = define_nn(2,2, layers, 1);
  printf("Number of inputs and outputs: %d %d\n", nn->inputs, nn->outputs);
  float input[2] = {2,3};
  float* output = forward_pass(nn, input);
  free_nn(nn);
  return 0;
}
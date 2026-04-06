#include <stdio.h>
#include "src/nn.cuh"

int main() {
  NN network;
  network.layers[0] = 784;
  network.layers[1] = 100
  network.layers[2] = 10;
  float *d_images, *d_labels;
  load_data(784, &d_images, &d_labels);

  init_nn(&network);

  free_nn(&network);
  free_input(d_images, d_labels);
  return 0;
}
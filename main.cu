#include <stdio.h>
#include "src/nn.cuh"

int main() {
  NN network;
  int layers[] = {784,100,10};
  float *d_images, *d_labels;
  load_data(784, &d_images, &d_labels);

  network.layers = layers;
  init_nn(&network);

  free_nn(&network);
  free_input(d_images, d_labels);
  return 0;
}

target:
	rm -fr build && mkdir build && cd build && cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc && cmake --build .


run:
	./build/CS4700

clean:
	rm -fr build
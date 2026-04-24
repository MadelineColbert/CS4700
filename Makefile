
target:
	rm -fr build && mkdir build && cd build && cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc && cmake --build .


run:
	./build/CS4700

clean:
	rm -fr build

mnist:
	wget --no-check-certificate https://raw.githubusercontent.com/fgnt/mnist/master/train-images-idx3-ubyte.gz
	wget --no-check-certificate https://raw.githubusercontent.com/fgnt/mnist/master/train-labels-idx1-ubyte.gz
	wget --no-check-certificate https://raw.githubusercontent.com/fgnt/mnist/master/t10k-images-idx3-ubyte.gz
	wget --no-check-certificate https://raw.githubusercontent.com/fgnt/mnist/master/t10k-labels-idx1-ubyte.gz
	gunzip train-images-idx3-ubyte.gz
	gunzip train-labels-idx1-ubyte.gz
	gunzip t10k-images-idx3-ubyte.gz
	gunzip t10k-labels-idx1-ubyte.gz


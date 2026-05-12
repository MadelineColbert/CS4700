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

EIGEN_VERSION = 3.4.0
EIGEN_DIR     = eigen
EIGEN_URL     = https://gitlab.com/libeigen/eigen/-/archive/$(EIGEN_VERSION)/eigen-$(EIGEN_VERSION).tar.gz

DENSE_SRC     = dense/densecpu.cpp
DENSE_TARGET  = dense/densecpu
CXX           = g++
CXXFLAGS      = -std=c++11 -I$(EIGEN_DIR) -O2

densecpu: $(EIGEN_DIR)
	$(CXX) $(CXXFLAGS) -o $(DENSE_TARGET) $(DENSE_SRC)

$(EIGEN_DIR):
	@echo "Downloading Eigen $(EIGEN_VERSION)..."
	mkdir -p $(EIGEN_DIR)
	@( wget -q -O - $(EIGEN_URL) 2>/dev/null || curl -sL $(EIGEN_URL) ) | tar -xz --strip-components=1 -C $(EIGEN_DIR)
	@if [ ! -f $(EIGEN_DIR)/Eigen/Core ]; then \
		echo "*** ERROR: Eigen download failed!"; \
		rm -rf $(EIGEN_DIR); \
		exit 1; \
	fi
	@echo "Eigen ready."

run-densecpu:
	cd dense && ./densecpu
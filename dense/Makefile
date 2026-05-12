# === Configuration ===
EIGEN_VERSION = 3.4.0
EIGEN_URL     = https://gitlab.com/libeigen/eigen/-/archive/$(EIGEN_VERSION)/eigen-$(EIGEN_VERSION).tar.gz
EIGEN_DIR     = eigen

SRC     = densecpu.cpp
TARGET  = densecpu
CXX     = g++
CXXFLAGS = -std=c++11 -I$(EIGEN_DIR) -O2

.PHONY: all clean distclean

all: $(TARGET)

$(EIGEN_DIR):
	@echo "Downloading Eigen $(EIGEN_VERSION)..."
	mkdir -p $(EIGEN_DIR)
	@( wget -q -O - $(EIGEN_URL) 2>/dev/null || curl -sL $(EIGEN_URL) ) | tar -xz --strip-components=1 -C $(EIGEN_DIR)
	@if [ ! -f $(EIGEN_DIR)/Eigen/Core ]; then \
		echo "*** ERROR: Eigen download failed!"; \
		rm -rf $(EIGEN_DIR); \
		exit 1; \
	fi
	@echo "Eigen downloaded successfully."

$(TARGET): $(SRC) $(EIGEN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC)

clean:
	rm -f $(TARGET)

distclean: clean
	rm -rf $(EIGEN_DIR)
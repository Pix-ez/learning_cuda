# Simple Makefile for CUDA project

NVCC        = nvcc
ARCH        = -arch=sm_75
SRC         = main.cu kernels.cu
TARGET      = benchmark
CFLAGS      = -O3

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(ARCH) $(CFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)

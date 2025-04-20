# Simple Makefile for CUDA project

NVCC        = nvcc
ARCH        = -arch=sm_75
SRC         = main.cu kernels.cu
TARGET      = benchmark
CFLAGS      = -O3
deps        = -lcublas -lcublasLt

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(ARCH) $(CFLAGS) $(deps) -o $@ $^

clean:
	rm -f $(TARGET)

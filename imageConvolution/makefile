# File: Makefile

# Compiler
NVCC = nvcc

# Target binary name
TARGET = imageConvolution

# Source files
SRCS = imageConvolutionv1.cu

# CUDA architecture (adjust if needed for your GPU)
ARCH = -gencode arch=compute_61,code=sm_61

# Includes (adjust if SDK helper headers are in a separate path)
INCLUDES = -I../Common



# Libraries
LIBS =

# Flags
NVCC_FLAGS = -O2 $(ARCH) $(INCLUDES)

# Default rule
all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Clean rule
clean:
	rm -f $(TARGET) *.o *.pgm *_out.pgm

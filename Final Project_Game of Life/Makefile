NVFLAGS := -std=c++11 -O3 -Xptxas=-v -arch=sm_61
LDFLAGS := -lpng -lz
TARGET := hw8

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	rm -f $(TARGET)

$(TARGET): hw8.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?



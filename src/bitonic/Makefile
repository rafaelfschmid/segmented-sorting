#TIME=1
#EXECS=10
#KERNELS=16
################################################################################
INCLUDES  := -I/usr/local/cuda-8.0/samples/common/inc
LIBRARIES := -lcuda
NVCC := nvcc
GCC_LIB_COMP := -ccbin g++ 
ALL_CCFLAGS := $(GCC_LIB_COMP) $(INCLUDES) -m64
GEN_CODE_FLAGS := -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -gencode arch=compute_61,code=compute_61

ALL_FLAGS := -std=c++11
VARIABLES := -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS) -DCONCURRENT_KERNELS=$(KERNELS)
################################################################################

# Target rules
all: build

build: bitonicsort.exe

bitonicsort.o: bitonicsort.cu
	$(NVCC) $(ALL_CCFLAGS) -dc $(GEN_CODE_FLAGS) -o $@ -c $<

main.o: main.cpp
	$(NVCC) $(ALL_CCFLAGS) -dc $(GEN_CODE_FLAGS) $(ALL_FLAGS) $(VARIABLES) -o $@ -c $<

bitonicsort.exe: bitonicsort.o main.o
	$(NVCC) $(GEN_CODE_FLAGS) -o bitonicsort$(KERNELS).exe $+

run: build
	./bitonicsort.exe

clean:
	rm -f bitonicsort.exe bitonicsort.o main.o

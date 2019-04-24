#TIME=1
#EXECS=10
################################################################################
INCLUDES  := -I/usr/local/cuda-7.5/samples/common/inc
LIBRARIES := -lcuda
NVCC := nvcc
GCC_LIB_COMP := -ccbin g++ 
ALL_CCFLAGS := $(GCC_LIB_COMP) $(INCLUDES) -m64
GEN_CODE_FLAGS := -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52
ALL_FLAGS := -std=c++11
VARIABLES := -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)
################################################################################

# Target rules
all: build

build: quicksort.exe

quicksort.o: quicksort.cu
	$(NVCC) $(ALL_CCFLAGS) -dc $(GEN_CODE_FLAGS) $(ALL_FLAGS) $(VARIABLES) -o $@ -c $<

bitonicsort.o: bitonicsort.cu
	$(NVCC) $(ALL_CCFLAGS) -dc $(GEN_CODE_FLAGS) $(VARIABLES) -o $@ -c $<

quicksort.exe: quicksort.o bitonicsort.o
	$(NVCC) $(GEN_CODE_FLAGS) -o $@ $+

run: build
	./quicksort.exe

clean:
	rm -f quicksort quicksort.o bitonicsort.o


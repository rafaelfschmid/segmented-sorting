#TIME=1
EXECS=1
STREAMS=8
################################################################################
INCLUDES  := -I$(CUDA_HOME)/samples/common/inc
LIBRARIES := -lcuda
NVCC := nvcc
GCC_LIB_COMP := -ccbin g++ -Xcompiler -fopenmp 
ALL_CCFLAGS := $(GCC_LIB_COMP) $(INCLUDES) -m64
GEN_CODE_FLAGS := -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52
ALL_FLAGS := -std=c++11
VARIABLES := -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS) -DNUM_STREAMS=$(STREAMS)
################################################################################

# Target rules
all: build

build: quicksort.exe quicksortthr.exe quicksortsched.exe quicksortschedthr.exe

quicksort.o: quicksort.cu
	$(NVCC) $(ALL_CCFLAGS) -dc $(GEN_CODE_FLAGS) $(ALL_FLAGS) $(VARIABLES) -o $@ -c $< -lcuda

quicksort.exe: quicksort.o 
	$(NVCC) $(GEN_CODE_FLAGS) -o quicksort$(STREAMS).exe $+ -lcuda

quicksortthr.exe: quicksort.o
	$(NVCC) $(GEN_CODE_FLAGS) -o quicksort$(STREAMS)thr.exe $+ -DTHREADS

quicksortsched.exe: quicksort.o
	$(NVCC) $(GEN_CODE_FLAGS) -o quicksort$(STREAMS)sched.exe $+ -DSCHEDULE

quicksortschedthr.exe: quicksort.o
	$(NVCC) $(GEN_CODE_FLAGS) -o quicksort$(STREAMS)schedthr.exe $+ -DSCHEDULE -DTHREADS

run: build
	./quicksort.exe

clean:
	rm -f quicksort quicksort.o bitonicsort.o


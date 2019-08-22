#TIME=1
#EXECS=10
STREAMS=1
################################################################################
INCLUDES  := -I$(CUDA_HOME)/samples/common/inc
LIBRARIES := -lcuda
NVCC := nvcc
GCC_LIB_COMP := -ccbin g++ 
ALL_CCFLAGS := $(GCC_LIB_COMP) $(INCLUDES) -m64
GEN_CODE_FLAGS := -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52
ALL_FLAGS := -std=c++11
VARIABLES := -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS) -DNUM_STREAMS=$(STREAMS)
################################################################################

# Target rules
#quicksort.o: quicksort.cu
#	$(NVCC) $(ALL_CCFLAGS) -dc $(GEN_CODE_FLAGS) $(ALL_FLAGS) $(VARIABLES) -o $@ -c $<

#quicksort.exe: quicksort.o 
#	$(NVCC) $(GEN_CODE_FLAGS) -o quicksort$(STREAMS).exe $+

#run: build
#	./quicksort.exe

#clean:
#	rm -f quicksort quicksort.o bitonicsort.o


# Target rules
all: build

build: mergeSort

check.deps:
ifeq ($(SAMPLE_ENABLED),0)
	@echo "Sample will be waived due to the above missing dependencies"
else
	@echo "Sample is ready - all dependencies have been met"
endif

bitonic.o:bitonic.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

main.o:main.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) $(ALL_FLAGS) $(VARIABLES) -o $@ -c $<

mergeSort.o:mergeSort.cu
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

mergeSort_host.o:mergeSort_host.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

mergeSort_validate.o:mergeSort_validate.cpp
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

mergeSort: bitonic.o main.o mergeSort.o mergeSort_host.o mergeSort_validate.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

# $(EXEC) mkdir -p ../../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)
# $(EXEC) cp $@ ../../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)

run: build
	$(EXEC) ./mergeSort

clean:
	rm -f mergeSort bitonic.o main.o mergeSort.o mergeSort_host.o mergeSort_validate.o
# rm -rf ../../bin/$(TARGET_ARCH)/$(TARGET_OS)/$(BUILD_TYPE)/mergeSort

clobber: clean

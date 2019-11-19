ARCH=61

NVFLAGS=-std=c++11 -gencode arch=compute_${ARCH},code=sm_${ARCH} -O3 -lcuda --expt-relaxed-constexpr -DEXECUTIONS=${EXECS} -DELAPSED_TIME=${TIME}

all: 
	nvcc $(NVFLAGS) main.cu -o bb_segsort.exe

clean:
	rm bb_segsort.exe


	

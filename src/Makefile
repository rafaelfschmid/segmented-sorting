TIME=1
BITS=12
EXECS=10
SEGS=1048576

arch=-gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52

all: mergeseg radixseg fixcub fixpass fixpassdiff fixthrust fixmergemgpu fixbitonic fixmerge fixoddeven fixquick bitonicseg 

mergeseg: mergeseg.cu
	nvcc $(arch) mergeseg.cu -o mergeseg.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

fixmergemgpu: fixmergemgpu.cu
	nvcc $(arch) fixmergemgpu.cu -o fixmergemgpu.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)


radixseg: radixseg.cu
	nvcc $(arch) radixseg.cu -o radixseg.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

fixcub: fixcub.cu
	nvcc $(arch) fixcub.cu -o fixcub.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)
 
fixpass: fixpass.cu
	nvcc -O3 $(arch) fixpass.cu -o fixpass.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

fixpassdiff: fixpassdiff.cu
	nvcc -O3 $(arch) fixpassdiff.cu -o fixpassdiff.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

fixthrust: fixthrust.cu
	nvcc $(arch) fixthrust.cu -o fixthrust.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

fixbitonic: fixsort/bitonic/main.cpp
	make clean -C fixsort/bitonic
	make -C fixsort/bitonic

bitonicseg: fixsort/bitonicseg/main.cpp
	make clean -C fixsort/bitonicseg
	make -C fixsort/bitonicseg

fixmerge: fixsort/mergesort/main.cpp
	make clean -C fixsort/mergesort
	make -C fixsort/mergesort

fixoddeven: fixsort/oddevensort/main.cpp
	make clean -C fixsort/oddevensort
	make -C fixsort/oddevensort

fixquick: fixsort/quicksort/quicksort.cu
	make clean -C fixsort/quicksort
	make -C fixsort/quicksort

nthrust: nthrust.cu
	nvcc $(arch) nthrust.cu -o nthrust.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)


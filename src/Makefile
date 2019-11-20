#TIME=1
#BITS=12
#EXECS=1
#STREAMS=32
ARQ=61

arch= -gencode arch=compute_${ARQ},code=sm_${ARQ} 

all: mergeseg radixseg fixcub fixthrust nthrust fixseq mseq

mergeseg: mergeseg.cu
	nvcc $(arch) mergeseg.cu -o mergeseg.exe -I"../../lib/moderngpu/src" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

radixseg: radixseg.cu
	nvcc $(arch) radixseg.cu -o radixseg.exe -I"../../lib/cub" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

fixcub: fixcub.cu
	nvcc $(arch) fixcub.cu -o fixcub.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)
 
fixthrust: fixthrust.cu
	nvcc $(arch) fixthrust.cu -o fixthrust.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

prevfixthrust: prevfixthrust.cu
	nvcc $(arch) prevfixthrust.cu -o prevfixthrust.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

fixpasscub: fixpasscub.cu
	nvcc $(arch) fixpasscub.cu -o fixpasscub.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)
 
fixpassthrust: fixpassthrust.cu
	nvcc $(arch) fixpassthrust.cu -o fixpassthrust.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

nthrust: nthrust.cu
	nvcc $(arch) nthrust.cu -o nthrust.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

fixseq: fixseq.cu
	nvcc $(arch) fixseq.cu -o fixseq.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

fixpassseq: fixpassseq.cu
	nvcc $(arch) fixpassseq.cu -o fixpassseq.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

mseq: mseq.cu
	nvcc $(arch) mseq.cu -o mseq.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

#############################################
cpu-gpu-thrust: cpu-gpu-thrust.cu
	nvcc -ccbin g++ -Xcompiler -fopenmp $(arch) cpu-gpu-thrust.cu -o cpu-gpu-thrust$(STREAMS).exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS) -DNUM_STREAMS=$(STREAMS)

newthrustsemaforo: newthrustsemaforo.cu
	nvcc -ccbin g++ -Xcompiler -fopenmp $(arch) newthrustsemaforo.cu -o newthrustsemaforo$(STREAMS).exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS) -DNUM_STREAMS=$(STREAMS)
############################################

newthrust: newthrust.cu
	nvcc -ccbin g++ -Xcompiler -fopenmp $(arch) newthrust.cu -o newthrust$(STREAMS).exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS) -DNUM_STREAMS=$(STREAMS)

newthrustthr: newthrust.cu
	nvcc -ccbin g++ -Xcompiler -fopenmp $(arch) newthrust.cu -o newthrust$(STREAMS)thr.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS) -DNUM_STREAMS=$(STREAMS) -DTHREADS

newthrustsched: newthrust.cu
	nvcc -ccbin g++ -Xcompiler -fopenmp $(arch) newthrust.cu -o newthrust$(STREAMS)sched.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS) -DNUM_STREAMS=$(STREAMS) -DSCHEDULE

newthrustschedthr: newthrust.cu
	nvcc -ccbin g++ -Xcompiler -fopenmp $(arch) newthrust.cu -o newthrust$(STREAMS)schedthr.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS) -DNUM_STREAMS=$(STREAMS) -DSCHEDULE -DTHREADS



newthrust1stream: newthrust1stream.cu
	nvcc $(arch) newthrust1stream.cu -o newthrust1stream.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

ncub: ncub.cu
	nvcc $(arch) ncub.cu -o ncub.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

nmoderngpu: nmoderngpu.cu
	nvcc $(arch) nmoderngpu.cu -o nmoderngpu.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

mergesort: mergesort.cu
	nvcc $(arch) mergesort.cu -o mergesort.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)
#for STREAMS in 1 8 16 32 ; do \
	#	nvcc $(arch) mergesort.cu -o mergesort$${STREAMS}.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS) -DNUM_STREAMS=$${STREAMS} ; \
	#done

simplemergesort: simplemergesort.cu
	nvcc $(arch) simplemergesort.cu -o simplemergesort.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS) -I/usr/local/cuda/samples/common/inc

#	for STREAMS in 1 8 16 32 ; do \
#	nvcc $(arch) mergesort.cu -o mergesort$${STREAMS}.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS) -DNUM_STREAMS=$${STREAMS} ; \
#	done


#TIME=1
#BITS=12
#EXECS=1
#STREAMS=32

arch= -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60

all: mergeseg radixseg fixcub fixthrust nthrust fixseq mseq

mergeseg: mergeseg.cu
	nvcc $(arch) mergeseg.cu -o mergeseg.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

radixseg: radixseg.cu
	nvcc $(arch) radixseg.cu -o radixseg.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

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

newthrust: newthrust.cu
	nvcc -ccbin g++ -Xcompiler -fopenmp $(arch) newthrust.cu -o newthrust$(STREAMS).exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS) -DNUM_STREAMS=$(STREAMS)

newthrust1stream: newthrust1stream.cu
	nvcc $(arch) newthrust1stream.cu -o newthrust1stream.exe -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

ncub: ncub.cu
	nvcc $(arch) ncub.cu -o ncub.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

nmoderngpu: nmoderngpu.cu
	nvcc $(arch) nmoderngpu.cu -o nmoderngpu.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS)

mergesort: mergesort.cu
	#for STREAMS in 1 8 16 32 ; do \
		nvcc $(arch) mergesort.cu -o mergesort$${STREAMS}.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS) -DNUM_STREAMS=$${STREAMS} ; \
	done
	nvcc $(arch) mergesort.cu -o mergesort.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS) ; \


simplemergesort: simplemergesort.cu
	nvcc $(arch) simplemergesort.cu -o simplemergesort.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS) -I/usr/local/cuda/samples/common/inc ; \

#	for STREAMS in 1 8 16 32 ; do \
#	nvcc $(arch) mergesort.cu -o mergesort$${STREAMS}.exe -I"../lib" -std=c++11 --expt-extended-lambda -lcuda -DELAPSED_TIME=$(TIME) -DEXECUTIONS=$(EXECS) -DNUM_STREAMS=$${STREAMS} ; \
#	done


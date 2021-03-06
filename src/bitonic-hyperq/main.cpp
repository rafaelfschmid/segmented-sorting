/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * This sample implements bitonic sort and odd-even merge sort, algorithms
 * belonging to the class of sorting networks.
 * While generally subefficient on large sequences
 * compared to algorithms with better asymptotic algorithmic complexity
 * (i.e. merge sort or radix sort), may be the algorithms of choice for sorting
 * batches of short- or mid-sized arrays.
 * Refer to the excellent tutorial by H. W. Lang:
 * http://www.iti.fh-flensburg.de/lang/algorithmen/sortieren/networks/indexen.htm
 *
 * Victor Podlozhnyuk, 07/09/2009
 */

// CUDA Runtime
#include <cuda_runtime.h>
#include <cuda.h>
#include "sortingNetworks_common.h"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#ifndef ELAPSED_TIME
#define ELAPSED_TIME 0
#endif

#ifndef CONCURRENT_KERNELS
#define CONCURRENT_KERNELS 16
#endif

void cudaTest(cudaError_t error) {
	if (error != cudaSuccess) {
		printf("cuda returned error %s (code %d), line(%d)\n",
				cudaGetErrorString(error), error, __LINE__);
		exit (EXIT_FAILURE);
	}
}

void print(uint* host_data, uint n) {
	std::cout << "\n";
	for (uint i = 0; i < n; i++) {
		std::cout << host_data[i] << " ";
	}
	std::cout << "\n";
}

int main(int argc, char** argv) {

	uint num_of_segments;
	uint num_of_elements;
	uint i;

	scanf("%d", &num_of_segments);
	uint mem_size_seg = sizeof(int) * (num_of_segments + 1);
	uint *h_seg = (uint *) malloc(mem_size_seg);
	for (i = 0; i < num_of_segments + 1; i++)
		scanf("%d", &h_seg[i]);

	scanf("%d", &num_of_elements);
	uint mem_size_vec = sizeof(int) * num_of_elements;
	uint *h_vec = (uint *) malloc(mem_size_vec);
	uint *h_value = (uint *) malloc(mem_size_vec);
	for (i = 0; i < num_of_elements; i++) {
		scanf("%d", &h_vec[i]);
		h_value[i] = i;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	uint *d_value, *d_value_out, *d_vec, *d_vec_out;

	cudaTest(cudaMalloc((void **) &d_vec, mem_size_vec));
	cudaTest(cudaMalloc((void **) &d_value, mem_size_vec));
	cudaTest(cudaMalloc((void **) &d_vec_out, mem_size_vec));
	cudaTest(cudaMalloc((void **) &d_value_out, mem_size_vec));

	int nStreams = CONCURRENT_KERNELS;
	if(CONCURRENT_KERNELS > num_of_segments)
		nStreams = num_of_segments;

	int elements_per_segment = num_of_elements/num_of_segments;

	// allocate and initialize an array of stream handles
	cudaStream_t *streams = (cudaStream_t *) malloc(nStreams * sizeof(cudaStream_t));
	for(int i = 0; i < nStreams; i++) {
			cudaStreamCreate(&streams[i]);
			//cudaStreamCreateWithFlags(&streams[i],cudaStreamNonBlocking);
		}

	for (int i = 0; i < EXECUTIONS; i++) {

		cudaTest(cudaMemcpy(d_vec, h_vec, mem_size_vec, cudaMemcpyHostToDevice));
		cudaTest(cudaMemcpy(d_value, h_value, mem_size_vec, cudaMemcpyHostToDevice));

		cudaEventRecord(start);
		if(elements_per_segment < 1024) {
						printf("########ERRROOORRRRR#######");
						printf("Number of elements per segment less than minimum.");
					//uint threadCount = bitonicSort(d_vec_out, d_value_out, d_vec, d_value,num_of_segments, elements_per_segment, 1, stream);
				}
		else {
			for (int j = 0; j < num_of_segments; j+=nStreams) {
				for (int k = 0; k < nStreams; k++) {
					uint threadCount = bitonicSort(d_vec_out+h_seg[j+k], d_value_out+h_seg[j+k], d_vec+h_seg[j+k], d_value+h_seg[j+k], 1, num_of_elements/num_of_segments, 1, streams[k]);
				}
			}
		}
		cudaEventRecord(stop);

		cudaError_t errSync = cudaGetLastError();
		cudaError_t errAsync = cudaDeviceSynchronize();
		if (errSync != cudaSuccess)
			printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
		if (errAsync != cudaSuccess)
			printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

		if (ELAPSED_TIME == 1) {
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			std::cout << milliseconds << "\n";
		}

		cudaDeviceSynchronize();
	}

	cudaMemcpy(h_vec, d_vec_out, mem_size_vec, cudaMemcpyDeviceToHost);

	cudaFree(d_vec);
	cudaFree(d_vec_out);
	cudaFree(d_value);
	cudaFree(d_value_out);

	if (ELAPSED_TIME != 1) {
		print(h_vec, elements_per_segment);
	}

	free(h_seg);
	free(h_vec);
	free(h_value);

	return 0;
}

////////////////////////////////////////////////////////////////////////////////
// Test driver
////////////////////////////////////////////////////////////////////////////////
/*int main(int argc, char **argv)
 {
 cudaError_t error;
 printf("%s Starting...\n\n", argv[0]);

 printf("Starting up CUDA context...\n");
 int dev = findCudaDevice(argc, (const char **)argv);

 uint *h_InputKey, *h_InputVal, *h_OutputKeyGPU, *h_OutputValGPU;
 uint *d_InputKey, *d_InputVal,    *d_OutputKey,    *d_OutputVal;
 StopWatchInterface *hTimer = NULL;

 const uint             N = 1048576;
 const uint           DIR = 0;
 const uint     numValues = 65536;
 const uint numIterations = 1;

 printf("Allocating and initializing host arrays...\n\n");
 sdkCreateTimer(&hTimer);
 h_InputKey     = (uint *)malloc(N * sizeof(uint));
 h_InputVal     = (uint *)malloc(N * sizeof(uint));
 h_OutputKeyGPU = (uint *)malloc(N * sizeof(uint));
 h_OutputValGPU = (uint *)malloc(N * sizeof(uint));
 srand(2001);

 for (uint i = 0; i < N; i++)
 {
 h_InputKey[i] = rand() % numValues;
 h_InputVal[i] = i;
 }

 printf("Allocating and initializing CUDA arrays...\n\n");
 error = cudaMalloc((void **)&d_InputKey,  N * sizeof(uint));
 checkCudaErrors(error);
 error = cudaMalloc((void **)&d_InputVal,  N * sizeof(uint));
 checkCudaErrors(error);
 error = cudaMalloc((void **)&d_OutputKey, N * sizeof(uint));
 checkCudaErrors(error);
 error = cudaMalloc((void **)&d_OutputVal, N * sizeof(uint));
 checkCudaErrors(error);
 error = cudaMemcpy(d_InputKey, h_InputKey, N * sizeof(uint), cudaMemcpyHostToDevice);
 checkCudaErrors(error);
 error = cudaMemcpy(d_InputVal, h_InputVal, N * sizeof(uint), cudaMemcpyHostToDevice);
 checkCudaErrors(error);

 int flag = 1;
 printf("Running GPU bitonic sort (%u identical iterations)...\n\n", numIterations);

 for (uint arrayLength = 64; arrayLength <= N; arrayLength *= 2)
 {
 printf("Testing array length %u (%u arrays per batch)...\n", arrayLength, N / arrayLength);
 error = cudaDeviceSynchronize();
 checkCudaErrors(error);

 sdkResetTimer(&hTimer);
 sdkStartTimer(&hTimer);
 uint threadCount = 0;

 for (uint i = 0; i < numIterations; i++)
 threadCount = bitonicSort(
 d_OutputKey,
 d_OutputVal,
 d_InputKey,
 d_InputVal,
 N / arrayLength,
 arrayLength,
 DIR
 );

 error = cudaDeviceSynchronize();
 checkCudaErrors(error);

 sdkStopTimer(&hTimer);
 printf("Average time: %f ms\n\n", sdkGetTimerValue(&hTimer) / numIterations);

 if (arrayLength == N)
 {
 double dTimeSecs = 1.0e-3 * sdkGetTimerValue(&hTimer) / numIterations;
 printf("sortingNetworks-bitonic, Throughput = %.4f MElements/s, Time = %.5f s, Size = %u elements, NumDevsUsed = %u, Workgroup = %u\n",
 (1.0e-6 * (double)arrayLength/dTimeSecs), dTimeSecs, arrayLength, 1, threadCount);
 }

 printf("\nValidating the results...\n");
 printf("...reading back GPU results\n");
 error = cudaMemcpy(h_OutputKeyGPU, d_OutputKey, N * sizeof(uint), cudaMemcpyDeviceToHost);
 checkCudaErrors(error);
 error = cudaMemcpy(h_OutputValGPU, d_OutputVal, N * sizeof(uint), cudaMemcpyDeviceToHost);
 checkCudaErrors(error);

 int keysFlag = validateSortedKeys(h_OutputKeyGPU, h_InputKey, N / arrayLength, arrayLength, numValues, DIR);
 int valuesFlag = validateValues(h_OutputKeyGPU, h_OutputValGPU, h_InputKey, N / arrayLength, arrayLength);
 flag = flag && keysFlag && valuesFlag;

 printf("\n");
 }

 printf("Shutting down...\n");
 sdkDeleteTimer(&hTimer);
 cudaFree(d_OutputVal);
 cudaFree(d_OutputKey);
 cudaFree(d_InputVal);
 cudaFree(d_InputKey);
 free(h_OutputValGPU);
 free(h_OutputKeyGPU);
 free(h_InputVal);
 free(h_InputKey);

 // cudaDeviceReset causes the driver to clean up all state. While
 // not mandatory in normal operation, it is good practice.  It is also
 // needed to ensure correct operation when the application is being
 // profiled. Calling cudaDeviceReset causes all profile data to be
 // flushed before the application exits
 cudaDeviceReset();
 exit(flag ? EXIT_SUCCESS : EXIT_FAILURE);
 }*/

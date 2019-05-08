/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "mergeSort_common.h"

#ifndef NUM_STREAMS
#define NUM_STREAMS 16
#endif

void print(uint* host_data, uint n, uint m) {
	std::cout << "\n";
	for (uint i = 0; i < n; i++) {
		for (uint j = 0; j < m; j++) {
			std::cout << host_data[i * m + j] << " ";
		}
		std::cout << "\n";
	}

}

////////////////////////////////////////////////////////////////////////////////
// Verify the results.
////////////////////////////////////////////////////////////////////////////////
void check_results(int n, int m, unsigned int *results_h)
{
    for (int i = 0 ; i < n ; ++i) {
    	for (uint j = 1; j < m; j++) {
    		if (results_h[i*m +j -1] > results_h[i*m +j])
			{
				std::cout << "Invalid item[" << j-1 << "]: " << results_h[i*m +j -1] << " greater than " << results_h[i*m +j] << std::endl;
				exit(EXIT_FAILURE);
			}
    	}
    }

    std::cout << "OK" << std::endl;
}
////////////////////////////////////////////////////////////////////////////////
// Test driver
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {

	uint num_of_segments;
	uint num_of_elements;

	scanf("%d", &num_of_segments);
	uint mem_size_seg = sizeof(uint) * (num_of_segments + 1);
	uint *h_seg = (uint *) malloc(mem_size_seg);
	for (int i = 0; i <= num_of_segments; i++) {
		scanf("%d", &h_seg[i]);
	}

	scanf("%d", &num_of_elements);
	uint mem_size_vec = sizeof(uint) * num_of_elements;
	uint *h_SrcKey = (uint *) malloc(mem_size_vec);
	uint *h_SrcVal = (uint *) malloc(mem_size_vec);
	for (int i = 0; i < num_of_elements; i++) {
		scanf("%d", &h_SrcKey[i]);
		h_SrcVal[i] = i;
	}

	cudaStream_t streams[NUM_STREAMS];
	for (int i = 0; i < NUM_STREAMS; i++) {
		cudaStreamCreate(&streams[i]);
	}

	int nstreams = NUM_STREAMS;
	if (NUM_STREAMS > num_of_segments)
		nstreams = num_of_segments;

	cudaEvent_t start, stop;
	const uint DIR = 1;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	uint *d_SrcKey, *d_SrcVal, *d_BufKey, *d_BufVal, *d_DstKey, *d_DstVal;

	checkCudaErrors(cudaMalloc((void **) &d_DstKey, mem_size_vec));
	checkCudaErrors(cudaMalloc((void **) &d_DstVal, mem_size_vec));
	checkCudaErrors(cudaMalloc((void **) &d_BufKey, mem_size_vec));
	checkCudaErrors(cudaMalloc((void **) &d_BufVal, mem_size_vec));
	checkCudaErrors(cudaMalloc((void **) &d_SrcKey, mem_size_vec));
	checkCudaErrors(cudaMalloc((void **) &d_SrcVal, mem_size_vec));

	for (int i = 0; i < EXECUTIONS; i++) {
		checkCudaErrors(cudaMemcpy(d_SrcKey, h_SrcKey, mem_size_vec, cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_SrcVal, h_SrcVal, mem_size_vec, cudaMemcpyHostToDevice));

		printf("Initializing GPU merge sort...\n");
		initMergeSort();

		printf("Running GPU merge sort...\n");
		cudaEventRecord(start);
		//for (int j = 0; j < num_of_segments; j += nstreams) {
//			for (int s = 0; s < nstreams; s++) {

//			}

		//}
		mergeSort(d_DstKey, d_DstVal, d_BufKey, d_BufVal, d_SrcKey, d_SrcVal, num_of_elements, DIR);
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

	printf("Reading back GPU merge sort results...\n");
	checkCudaErrors(cudaMemcpy(h_SrcKey, d_DstKey, mem_size_vec,cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_SrcVal, d_DstVal, mem_size_vec,cudaMemcpyDeviceToHost));

	if (ELAPSED_TIME != 1) {
		print(h_SrcKey, num_of_segments, num_of_elements/num_of_segments);
		check_results(num_of_segments, num_of_elements/num_of_segments, h_SrcKey);
	}

	//printf("Inspecting the results...\n");
	//uint keysFlag = validateSortedKeys(h_DstKey, h_SrcKey, 1, N, numValues, DIR);
	//uint valuesFlag = validateSortedValues(h_DstKey, h_DstVal, h_SrcKey, 1, N);

	printf("Shutting down...\n");
	closeMergeSort();
	checkCudaErrors(cudaFree(d_SrcVal));
	checkCudaErrors(cudaFree(d_SrcKey));
	checkCudaErrors(cudaFree(d_BufVal));
	checkCudaErrors(cudaFree(d_BufKey));
	checkCudaErrors(cudaFree(d_DstVal));
	checkCudaErrors(cudaFree(d_DstKey));
	//free(h_DstVal);
	//free(h_DstKey);
	free(h_SrcVal);
	free(h_SrcKey);
}

/*uint *h_SrcKey, *h_SrcVal, *h_DstKey, *h_DstVal;
uint *d_SrcKey, *d_SrcVal, *d_BufKey, *d_BufVal, *d_DstKey, *d_DstVal;
StopWatchInterface *hTimer = NULL;

const uint N = 4 * 1048576;
const uint DIR = 1;
const uint numValues = 65536;

printf("%s Starting...\n\n", argv[0]);

int dev = findCudaDevice(argc, (const char **) argv);

if (dev == -1) {
	return EXIT_FAILURE;
}

printf("Allocating and initializing host arrays...\n\n");
sdkCreateTimer(&hTimer);
h_SrcKey = (uint *) malloc(N * sizeof(uint));
h_SrcVal = (uint *) malloc(N * sizeof(uint));
h_DstKey = (uint *) malloc(N * sizeof(uint));
h_DstVal = (uint *) malloc(N * sizeof(uint));

srand(2009);

for (uint i = 0; i < N; i++) {
	h_SrcKey[i] = rand() % numValues;
}

fillValues(h_SrcVal, N);

printf("Allocating and initializing CUDA arrays...\n\n");
checkCudaErrors(cudaMalloc((void **) &d_DstKey, N * sizeof(uint)));
checkCudaErrors(cudaMalloc((void **) &d_DstVal, N * sizeof(uint)));
checkCudaErrors(cudaMalloc((void **) &d_BufKey, N * sizeof(uint)));
checkCudaErrors(cudaMalloc((void **) &d_BufVal, N * sizeof(uint)));
checkCudaErrors(cudaMalloc((void **) &d_SrcKey, N * sizeof(uint)));
checkCudaErrors(cudaMalloc((void **) &d_SrcVal, N * sizeof(uint)));
checkCudaErrors(
		cudaMemcpy(d_SrcKey, h_SrcKey, N * sizeof(uint),
				cudaMemcpyHostToDevice));
checkCudaErrors(
		cudaMemcpy(d_SrcVal, h_SrcVal, N * sizeof(uint),
				cudaMemcpyHostToDevice));

printf("Initializing GPU merge sort...\n");
initMergeSort();

printf("Running GPU merge sort...\n");
checkCudaErrors(cudaDeviceSynchronize());
sdkResetTimer(&hTimer);
sdkStartTimer(&hTimer);
mergeSort(d_DstKey, d_DstVal, d_BufKey, d_BufVal, d_SrcKey, d_SrcVal, N, DIR);
checkCudaErrors(cudaDeviceSynchronize());
sdkStopTimer(&hTimer);
printf("Time: %f ms\n", sdkGetTimerValue(&hTimer));

printf("Reading back GPU merge sort results...\n");
checkCudaErrors(
		cudaMemcpy(h_DstKey, d_DstKey, N * sizeof(uint),
				cudaMemcpyDeviceToHost));
checkCudaErrors(
		cudaMemcpy(h_DstVal, d_DstVal, N * sizeof(uint),
				cudaMemcpyDeviceToHost));

printf("Inspecting the results...\n");
uint keysFlag = validateSortedKeys(h_DstKey, h_SrcKey, 1, N, numValues, DIR);

uint valuesFlag = validateSortedValues(h_DstKey, h_DstVal, h_SrcKey, 1, N);

printf("Shutting down...\n");
closeMergeSort();
sdkDeleteTimer(&hTimer);
checkCudaErrors(cudaFree(d_SrcVal));
checkCudaErrors(cudaFree(d_SrcKey));
checkCudaErrors(cudaFree(d_BufVal));
checkCudaErrors(cudaFree(d_BufKey));
checkCudaErrors(cudaFree(d_DstVal));
checkCudaErrors(cudaFree(d_DstKey));
free(h_DstVal);
free(h_DstKey);
free(h_SrcVal);
free(h_SrcKey);

exit((keysFlag && valuesFlag) ? EXIT_SUCCESS : EXIT_FAILURE);
}*/

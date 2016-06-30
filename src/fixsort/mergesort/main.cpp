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
#include <chrono>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "mergeSort_common.h"
#include <math.h>

#ifndef ELAPSED_TIME
#define ELAPSED_TIME 0
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
////////////////////////////////////////////////////////////////////////////////
// Test driver
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
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

	uint maxValue = 0;
	for (i = 0; i < num_of_elements; i++) {
		if(maxValue < h_vec[i])
			maxValue = h_vec[i];
	}

	uint mostSignificantBit = (uint)log2((double)maxValue) + 1;

	for (i = 0; i < num_of_segments; i++) {
		for (uint j = h_seg[i]; j < h_seg[i + 1]; j++) {
			uint segIndex = i << mostSignificantBit;
			h_vec[j] += segIndex;
		}
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	uint *d_value, *d_value_out, *d_vec, *d_vec_out;
	uint *d_BufKey, *d_BufVal;

	cudaTest(cudaMalloc((void **) &d_vec, mem_size_vec));
	cudaTest(cudaMalloc((void **) &d_value, mem_size_vec));
	cudaTest(cudaMalloc((void **) &d_vec_out, mem_size_vec));
	cudaTest(cudaMalloc((void **) &d_value_out, mem_size_vec));

	cudaTest(cudaMalloc((void **) &d_BufKey, mem_size_vec));
	cudaTest(cudaMalloc((void **) &d_BufVal, mem_size_vec));

	for (int i = 0; i < EXECUTIONS; i++) {

		cudaTest(cudaMemcpy(d_vec, h_vec, mem_size_vec, cudaMemcpyHostToDevice));
		cudaTest(cudaMemcpy(d_value, h_value, mem_size_vec, cudaMemcpyHostToDevice));

		cudaEventRecord(start);
		initMergeSort();
		mergeSort(d_vec_out, d_value_out, d_BufKey, d_BufVal, d_vec, d_value, num_of_elements, 1);
		closeMergeSort();
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

	for (i = 0; i < num_of_segments; i++) {
		for (uint j = h_seg[i]; j < h_seg[i + 1]; j++) {
			uint segIndex = i << mostSignificantBit;
			h_vec[j] -= segIndex;
		}
	}

	cudaFree(d_vec);
	cudaFree(d_vec_out);
	cudaFree(d_value);
	cudaFree(d_value_out);
	cudaFree(d_BufVal);
	cudaFree(d_BufKey);

	if (ELAPSED_TIME != 1) {
		print(h_vec, num_of_elements);
	}

	free(h_seg);
	free(h_vec);
	free(h_value);
}

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

#include <moderngpu/kernel_mergesort.hxx>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <utility>
#include <iostream>
#include <bitset>
#include <math.h>
#include <time.h>
#include <chrono>
#include <cuda.h>
//#include <cstdlib>
#include <iostream>
#include <chrono>

#ifndef ELAPSED_TIME
#define ELAPSED_TIME 0
#endif

#ifndef EXECUTIONS
#define EXECUTIONS 10
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
	uint mem_size_seg = sizeof(uint) * (num_of_segments + 1);
	uint *h_seg = (uint *) malloc(mem_size_seg);
	for (i = 0; i < num_of_segments + 1; i++)
		scanf("%d", &h_seg[i]);

	scanf("%d", &num_of_elements);
	uint mem_size_vec = sizeof(uint) * num_of_elements;
	uint *h_vec = (uint *) malloc(mem_size_vec);
	for (i = 0; i < num_of_elements; i++) {
		scanf("%d", &h_vec[i]);
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

	uint *d_vec;

	cudaTest(cudaMalloc((void **) &d_vec, mem_size_vec));

	for (int i = 0; i < EXECUTIONS; i++) {

		cudaTest(cudaMemcpy(d_vec, h_vec, mem_size_vec, cudaMemcpyHostToDevice));
		try {
			cudaEventRecord(start);
			mgpu::standard_context_t context;
			mgpu::mergesort(d_vec, num_of_elements, mgpu::less_t<uint>(), context);
			cudaEventRecord(stop);
		} catch (mgpu::cuda_exception_t ex) {
				cudaError_t errSync = cudaGetLastError();
				cudaError_t errAsync = cudaDeviceSynchronize();
				if (errSync != cudaSuccess)
						printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
				if (errAsync != cudaSuccess)
						printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
		}

		if (ELAPSED_TIME == 1) {
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			std::cout << milliseconds << "\n";
		}

		cudaDeviceSynchronize();
	}

	cudaMemcpy(h_vec, d_vec, mem_size_vec, cudaMemcpyDeviceToHost);

	for (i = 0; i < num_of_segments; i++) {
		for (uint j = h_seg[i]; j < h_seg[i + 1]; j++) {
			uint segIndex = i << mostSignificantBit;
			h_vec[j] -= segIndex;
		}
	}

	cudaFree(d_vec);

	if (ELAPSED_TIME != 1) {
//		print(h_vec, num_of_elements);
		print(h_vec, 30);

	}

	free(h_seg);
	free(h_vec);
}

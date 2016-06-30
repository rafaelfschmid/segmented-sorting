/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
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

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 *
 * See cuda.h for error code descriptions.
 */

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

#ifndef ELAPSED_TIME
#define ELAPSED_TIME 0
#endif

#ifndef EXP_BITS_SIZE
#define EXP_BITS_SIZE 10
#endif

void print(int* h_data, int n) {
	std::cout << "\n";
	for (int i = 0; i < n; i++) {
		std::cout << h_data[i] << " ";
	}
	std::cout << "\n";
}

void cudaTest(cudaError_t error) {
	if (error != cudaSuccess) {
		printf("cuda returned error %s (code %d), line(%d)\n",
				cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}
}

template<int BITS_NUMBER = 64>
__global__ void radix_sort(int *d_vec, int *d_aux, int *d_seg,
		int num_segments) {

	int bx = blockIdx.x;
	int tx = threadIdx.x;

	if (bx * blockDim.x + tx < num_segments) {
		int begin = d_seg[bx * blockDim.x + tx];
		int end = d_seg[bx * blockDim.x + tx + 1];

		int i, exp = 0;

		for (exp = 0; exp < BITS_NUMBER; exp++) {
			int one = 0;
			int zero = 0;

			for (i = begin; i < end; i++) {
				int x = (d_vec[i] >> exp) & 1;
				one += x;
				zero += (1 - x);
			}

			one += zero;

			for (i = end - 1; i >= begin; i--) {
				int x = (d_vec[i] >> exp) & 1;
				int index = begin + x * (one - 1) + (1 - x) * (zero - 1);
				d_aux[index] = d_vec[i];

				one -= x;
				zero -= (1 - x);
			}

			for (i = begin; i < end; i++)
				d_vec[i] = d_aux[i];
		}
	}
}

int main(int argc, char **argv) {

	int num_of_segments;
	int num_of_elements;
	int i;

	scanf("%d", &num_of_segments);
	int mem_size_seg = sizeof(int) * (num_of_segments + 1);
	int *h_seg = (int *) malloc(mem_size_seg);
	for (i = 0; i < num_of_segments + 1; i++)
		scanf("%d", &h_seg[i]);

	scanf("%d", &num_of_elements);
	int mem_size_vec = sizeof(int) * num_of_elements;
	int *h_vec = (int *) malloc(mem_size_vec);
	for (i = 0; i < num_of_elements; i++)
		scanf("%d", &h_vec[i]);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Allocate device memory
	int *d_seg, *d_vec, *d_aux;

	cudaTest(cudaMalloc((void **) &d_seg, mem_size_seg));
	cudaTest(cudaMalloc((void **) &d_vec, mem_size_vec));
	cudaTest(cudaMalloc((void **) &d_aux, mem_size_vec));

	cudaTest(cudaMemcpy(d_seg, h_seg, mem_size_seg, cudaMemcpyHostToDevice));
	cudaTest(cudaMemcpy(d_vec, h_vec, mem_size_vec, cudaMemcpyHostToDevice));

	int blocksize = BLOCK_SIZE;
	dim3 threads(blocksize, 1);
	dim3 grid((num_of_segments - 1) / blocksize + 1, 1);

	cudaEventRecord(start);
	radix_sort<EXP_BITS_SIZE> <<<grid, threads>>>(d_vec, d_aux, d_seg,
			num_of_segments);
	cudaEventRecord(stop);

	//cudaDeviceSynchronize();
	//cudaTest(cudaPeekAtLastError());
	cudaTest(cudaMemcpy(h_seg, d_seg, mem_size_seg, cudaMemcpyDeviceToHost));
	cudaTest(cudaMemcpy(h_vec, d_vec, mem_size_vec, cudaMemcpyDeviceToHost));

	if (ELAPSED_TIME == 1) {
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		std::cout << milliseconds << "\n";
	} else
		print(h_vec, num_of_elements);

	free(h_seg);
	free(h_vec);
	cudaFree(d_seg);
	cudaFree(d_vec);
	cudaFree(d_aux);
	//cudaDeviceReset();

	return 0;
}

/**
 * 	// cudaDeviceReset causes the driver to clean up all state. While
 // not mandatory in normal operation, it is good practice.  It is also
 // needed to ensure correct operation when the application is being
 // profiled. Calling cudaDeviceReset causes all profile data to be
 // flushed before the application exits
 cudaDeviceReset();
 */

/*
 printf("thread=%d | aux=%d %d %d %d\n", bx * blockDim.x + tx, d_aux[begin], d_aux[begin + 1], d_aux[begin + 2],	d_aux[begin + 3]);
 int devID = 0;
 cudaDeviceProp deviceProp;
 cudaTest(cudaGetDeviceProperties(&deviceProp, devID));
 unsigned int multiprocessorNumber = deviceProp.multiProcessorCount;
 unsigned int sharedMemoryTotal = deviceProp.sharedMemPerBlock/(sizeof(int));
 std::cout << "multiprocessorNumber: " << multiprocessorNumber << "\n";
 std::cout << "sharedMemoryTotal: " << sharedMemoryTotal << "\n";
 std::cout << "numberOfSegmentsPerBlock: " << sharedMemoryTotal << "\n";
 */

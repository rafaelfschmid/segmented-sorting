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
#include <iostream>

#include <cuda.h>

// Thread block size
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif

#ifndef SEGMENT_SIZE
#define SEGMENT_SIZE 2
#endif

void cudaTest(cudaError_t error) {
	if (error != cudaSuccess) {
		printf("cuda returned error %s (code %d), line(%d)\n",
				cudaGetErrorString(error), error, __LINE__);
		exit (EXIT_FAILURE);
	}
}

void print(uint* host_data, uint n, uint m) {
	std::cout << "\n";
	for (uint i = 0; i < n; i++) {
		for (uint j = 0; j < m; j++) {
			std::cout << host_data[i * n + j] << "\t";
		}
		std::cout << "\n";
	}

}

//__global__ void bitonic_sort_step(uint *dev_values, int k, int p, int n) {
__global__ void block_sorting(uint *d_vec, int n) {

	uint j = blockDim.x * blockIdx.x + threadIdx.x;

	if(j < n) {
		__shared__ uint As[BLOCK_SIZE];
		As[threadIdx.x] = d_vec[j];
		uint i = threadIdx.x;
		__syncthreads();

		for (int k = 2; k <= BLOCK_SIZE && k <= n; k <<= 1) { // sorting only block size row

			for (int p = k >> 1; p > 0; p = p >> 1) {

				uint ixp = i ^ p;

				/* The threads with the lowest ids sort the array. */
				if (i < ixp) {

					bool up = ((i & k) == 0); // sorting only block size matrix row

					// Sort ascending or descending according to up value
					if ((As[i] > As[ixp]) == up) {
						// exchange(i,ixj);
						uint temp = As[i];
						As[i] = As[ixp];
						As[ixp] = temp;
					}

				}

				__syncthreads();
			}
		}

		d_vec[j] = As[threadIdx.x];
	}
}

//__global__ void bitonic_sort_step(uint *dev_values, int k, int p, int n) {
__global__ void merge_blocks(uint *d_vec, int n) {

	uint i = blockDim.x * blockIdx.x + threadIdx.x;

	uint initial = BLOCK_SIZE;// << 1;
	for (int k = initial; k <= n; k <<= 1) { // sorting only block size row

		for (int p = k >> 1; p > 0; p = p >> 1) {

			uint ixp = i ^ p;

			/* The threads with the lowest ids sort the array. */
			if (i < ixp) {

				bool up = ((threadIdx.x & k) == 0); // sorting only block size matrix row

				// Sort ascending or descending according to up value
				if ((d_vec[i] > d_vec[ixp]) == up) {
					// exchange(i,ixj);
					uint temp = d_vec[i];
					d_vec[i] = d_vec[ixp];
					d_vec[ixp] = temp;
				}

			}

			__syncthreads();
		}
	}
}

int main(int argc, char** argv) {
	uint num_of_segments;
	uint num_of_elements;
	uint i;

	printf("teste0\n");
	scanf("%d", &num_of_segments);
	uint mem_size_seg = sizeof(uint) * (num_of_segments+1);
	uint *h_seg = (uint *) malloc(mem_size_seg);
	for (i = 0; i <= num_of_segments; i++)
		scanf("%d", &h_seg[i]);

	printf("segments=%d\n", num_of_segments);
	scanf("%d", &num_of_elements);
	uint mem_size_vec = sizeof(uint) * num_of_elements;
	uint *h_vec = (uint *) malloc(mem_size_vec);
	for (i = 0; i < num_of_elements; i++)
		scanf("%d", &h_vec[i]);

	printf("elements=%d\n", num_of_elements);
	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	uint *d_vec;

	cudaTest(cudaMalloc((void **) &d_vec, mem_size_vec));
	uint elements_per_segment = num_of_elements/num_of_segments;
	printf("elements_per_segment=%d\n", elements_per_segment);

	for (int i = 0; i < EXECUTIONS; i++) {

		cudaTest(cudaMemcpy(d_vec, h_vec, mem_size_vec, cudaMemcpyHostToDevice));

		dim3 dimBlock(BLOCK_SIZE, 1);
		dim3 dimGrid((elements_per_segment-1) / dimBlock.x + 1, 1);

		cudaEventRecord(start);
		for(int j = 0; j < 1; j++){ //num_of_segments; j++) {
			block_sorting<<<dimGrid, dimBlock>>>(d_vec+h_seg[j], elements_per_segment);

			cudaMemcpy(h_vec, d_vec, mem_size_vec, cudaMemcpyDeviceToHost);
			print(h_vec, num_of_segments, elements_per_segment);

			if(elements_per_segment > BLOCK_SIZE)
				merge_blocks <<<dimGrid, dimBlock>>>(d_vec+h_seg[j], elements_per_segment);

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

	cudaMemcpy(h_vec, d_vec, mem_size_vec, cudaMemcpyDeviceToHost);

	cudaFree(d_vec);

	if (ELAPSED_TIME != 1) {
		print(h_vec, num_of_segments, elements_per_segment);
	}

	free(h_vec);

	return 0;
}

/*
 * for (int p = 0; p < logn; p++) {
 for (int q = 0; q <= p; q++) {

 int d = 1 << (p-q);
 //for(int i = 0; i < n; i++) {
 bool up = ((col >> p) & 2) == 0;

 if ((col & d) == 0 && (As[row][col] > As[row][col | d]) == up) {
 int t = As[row][col];
 As[row][col] = As[row][col | d];
 As[row][col | d] = t;
 }
 //			}
 }
 }
 */

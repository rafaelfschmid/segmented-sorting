#include <iostream>
#include "helper_cuda.h"
#include <sys/time.h>

/**
 * mergesort.cu
 * a one-file c++ / cuda program for performing mergesort on the GPU
 * While the program execution is fairly slow, most of its runnning time
 *  is spent allocating memory on the GPU.
 * For a more complex program that performs many calculations,
 *  running on the GPU may provide a significant boost in performance
 */

// data[], size, threads, blocks, 
void mergesort(uint*, uint, dim3, dim3);
// A[]. B[], size, width, slices, nThreads
__global__ void gpu_mergesort(uint*, uint*, uint, uint, uint, dim3*, dim3*);
__device__ void gpu_bottomUpMerge(uint*, uint*, uint, uint, uint);

/*
 ============================================================================
 Name        : sorting_segments.cu
 Author      : Rafael Schmid
 Version     :
 Copyright   : Your copyright notice
 Description : Compute sum of reciprocals using STL on CPU and Thrust on GPU
 ============================================================================

 COMPILAR USANDO O SEGUINTE COMANDO:

 nvcc segmented_sort.cu -o segmented_sort -std=c++11 --expt-extended-lambda -I"/home/schmid/Dropbox/Unicamp/workspace/sorting_segments/moderngpu-master/src"

 */
#include <moderngpu/kernel_mergesort.hxx>

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <utility>
#include <iostream>
#include <bitset>
#include <math.h>

#include <cuda.h>
//#include <cstdlib>
#include <iostream>

#include <future>
#include<tuple>
#include <vector>

#ifndef ELAPSED_TIME
#define ELAPSED_TIME 0
#endif

#ifndef NUM_STREAMS
#define NUM_STREAMS 1
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
void check_results(int n, int m, unsigned int *results_h) {
	for (int i = 0; i < n; ++i) {
		for (uint j = 1; j < m; j++) {
			if (results_h[i * m + j - 1] > results_h[i * m + j]) {
				std::cout << "Invalid item[" << j - 1 << "]: "
						<< results_h[i * m + j - 1] << " greater than "
						<< results_h[i * m + j] << std::endl;
				exit (EXIT_FAILURE);
			}
		}
	}

	std::cout << "OK" << std::endl;
}

int main(int argc, char** argv) {

	uint num_of_segments;
	uint num_of_elements;
	uint i;

	scanf("%d", &num_of_segments);
	uint mem_size_seg = sizeof(uint) * (num_of_segments);
	uint *h_seg = (uint *) malloc(mem_size_seg);
	for (i = 0; i < num_of_segments + 1; i++)
		scanf("%d", &h_seg[i]);

	scanf("%d", &num_of_elements);
	uint mem_size_vec = sizeof(uint) * num_of_elements;
	uint *h_vec = (uint *) malloc(mem_size_vec);
	for (i = 0; i < num_of_elements; i++)
		scanf("%d", &h_vec[i]);

	dim3 threadsPerBlock;
	dim3 blocksPerGrid;

	threadsPerBlock.x = 1024;
	threadsPerBlock.y = 1;
	threadsPerBlock.z = 1;

	if((num_of_elements/num_of_segments) < 1024)
		threadsPerBlock.x = (num_of_elements/num_of_segments);

	blocksPerGrid.x = (num_of_elements/num_of_segments)/threadsPerBlock.x;
	blocksPerGrid.y = 1;
	blocksPerGrid.z = 1;

	int nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z
			* blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	uint *d_vec, *d_swp;
	dim3 *D_threads, *D_blocks;

	checkCudaErrors(cudaMalloc((void**) &D_threads, sizeof(dim3)));
	checkCudaErrors(cudaMalloc((void**) &D_blocks, sizeof(dim3)));

	checkCudaErrors(cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice));

	cudaStream_t streams[NUM_STREAMS];
	for (int i = 0; i < NUM_STREAMS; i++) {
		cudaStreamCreate(&streams[i]);
	}

	int nstreams = NUM_STREAMS;
	if (NUM_STREAMS > num_of_segments)
		nstreams = num_of_segments;

	checkCudaErrors(cudaMalloc((void **) &d_vec, mem_size_vec));
	checkCudaErrors(cudaMalloc((void **) &d_swp, mem_size_vec));

	for (uint j = 0; j < EXECUTIONS; j++) {

		// copy host memory to device
		checkCudaErrors(cudaMemcpy(d_vec, h_vec, mem_size_vec, cudaMemcpyHostToDevice));


		cudaEventRecord(start);
		for (int i = 0; i < num_of_segments; i += nstreams) {
			for (int s = 0; s < nstreams; s++) {
				for (int width = 2; width < (num_of_elements << 1);
						width <<= 1) {
					long slices = num_of_elements / ((nThreads) * width) + 1;
					// Actually call the kernel
					gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(d_vec,
							d_swp, num_of_elements, width, slices,
							D_threads, D_blocks);
				}
			}
		}
		cudaEventRecord(stop);


		cudaError_t errSync = cudaGetLastError();
		cudaError_t errAsync = cudaDeviceSynchronize();
		if (errSync != cudaSuccess)
			printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
		if (errAsync != cudaSuccess)
			printf("Async kernel error: %s\n",
					cudaGetErrorString(errAsync));


		if (ELAPSED_TIME == 1) {
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			std::cout << milliseconds << "\n";
		}

		cudaDeviceSynchronize();
	}

	checkCudaErrors(cudaMemcpy(h_vec, d_vec, mem_size_vec, cudaMemcpyDeviceToHost));

	cudaFree (d_swp);
	cudaFree(d_vec);
	cudaFree(D_blocks);
	cudaFree(D_threads);

	if (ELAPSED_TIME != 1) {
		print(h_vec, num_of_segments, num_of_elements/num_of_segments);
		check_results(num_of_segments, num_of_elements / num_of_segments,
				h_vec);
	}

	free(h_seg);
	free(h_vec);

	return 0;
}

// GPU helper function
// calculate the id of the current thread
__device__ unsigned int getIdx(dim3* threads, dim3* blocks) {
	int x;
	return threadIdx.x + threadIdx.y * (x = threads->x) + threadIdx.z * (x *=
			threads->y) + blockIdx.x * (x *= threads->z) + blockIdx.y * (x *=
			blocks->z) + blockIdx.z * (x *= blocks->y);
}

//
// Perform a full mergesort on our section of the data.
//
__global__ void gpu_mergesort(uint* source, uint* dest, uint size, uint width,
		uint slices, dim3* threads, dim3* blocks) {
	unsigned int idx = getIdx(threads, blocks);
	uint start = width * idx * slices, middle, end;

	for (uint slice = 0; slice < slices; slice++) {
		if (start >= size)
			break;

		middle = min(start + (width >> 1), size);
		end = min(start + width, size);
		gpu_bottomUpMerge(source, dest, start, middle, end);
		start += width;
	}
}

//
// Finally, sort something
// gets called by gpu_mergesort() for each slice
//
__device__ void gpu_bottomUpMerge(uint* source, uint* dest, uint start,
		uint middle, uint end) {
	uint i = start;
	uint j = middle;
	for (uint k = start; k < end; k++) {
		if (i < middle && (j >= end || source[i] < source[j])) {
			dest[k] = source[i];
			i++;
		} else {
			dest[k] = source[j];
			j++;
		}
	}
}


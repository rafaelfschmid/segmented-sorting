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

#include <moderngpu/kernel_segsort.hxx>

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

using namespace std::chrono;

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

void printSeg(uint* host_data, uint num_seg, uint num_ele) {
	std::cout << "\n";
	for (uint i = 0; i < num_seg; i++) {
		std::cout << host_data[i] << " ";
	}
	std::cout << num_ele << " ";
	std::cout << "\n";
}

int main(int argc, char** argv) {

	uint num_of_segments;
	uint num_of_elements;
	uint i;

	scanf("%d", &num_of_segments);
	uint mem_size_seg = sizeof(uint) * (num_of_segments);
	uint *h_seg = (uint *) malloc(mem_size_seg);
	for (i = 0; i < num_of_segments; i++)
		scanf("%d", &h_seg[i]);

	/*
	 * Different from other algorithms this do not need the last segment element
	 */
	uint aux;
	scanf("%d", &aux);

	scanf("%d", &num_of_elements);
	uint mem_size_vec = sizeof(uint) * num_of_elements;
	uint *h_vec = (uint *) malloc(mem_size_vec);
	for (i = 0; i < num_of_elements; i++)
		scanf("%d", &h_vec[i]);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	uint *d_seg, *d_vec, *d_index_resp;

	cudaTest(cudaMalloc((void **) &d_seg, mem_size_seg));
	cudaTest(cudaMalloc((void **) &d_vec, mem_size_vec));
	cudaTest(cudaMalloc((void **) &d_index_resp, mem_size_vec));

	for (uint j = 0; j < EXECUTIONS; j++) {

		// copy host memory to device
		cudaTest(cudaMemcpy(d_seg, h_seg, mem_size_seg, cudaMemcpyHostToDevice));
		cudaTest(cudaMemcpy(d_vec, h_vec, mem_size_vec, cudaMemcpyHostToDevice));

		try {
		cudaEventRecord(start);
		mgpu::standard_context_t context;
		mgpu::segmented_sort(d_vec, d_index_resp, num_of_elements, d_seg,
				num_of_segments, mgpu::less_t<uint>(), context);
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

	cudaTest(cudaMemcpy(h_vec, d_vec, mem_size_vec, cudaMemcpyDeviceToHost));

	cudaFree(d_seg);
	cudaFree(d_vec);
	cudaFree(d_index_resp);

	if (ELAPSED_TIME != 1) {
		print(h_vec, num_of_elements);
	}

	free(h_seg);
	free(h_vec);

	return 0;
}

/***
 * SEGMENTED SORT FUNCIONANDO
 *
 *
 uint n = atoi(argv[1]);
 uint m = atoi(argv[2]);
 uint num_segments = n / m;
 mgpu::standard_context_t context;
 rand_key<uint> func(m);

 mgpu::mem_t<uint> segs = mgpu::fill_function(func, num_segments, context);
 //mgpu::mem_t<uint> segs = mgpu::fill_random(0, n - 1, num_segments, true, context);
 std::vector<uint> segs_host = mgpu::from_mem(segs);
 mgpu::mem_t<uint> data = mgpu::fill_random(0, pow(2, NUMBER_BITS_SIZE), n,
 false, context);
 mgpu::mem_t<uint> values(n, context);
 std::vector<uint> data_host = mgpu::from_mem(data);

 //	print(segs_host); print(data_host);

 mgpu::segmented_sort(data.data(), values.data(), n, segs.data(),
 num_segments, mgpu::less_t<uint>(), context);

 std::vector<uint> sorted = from_mem(data);
 std::vector<uint> indices_host = from_mem(values);

 std::cout << "\n";
 //print(segs_host);
 //	print(data_host); print(indices_host);
 *
 */
